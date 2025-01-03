use bevy::prelude::*;
use bevy::utils::tracing;
use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, input::common_conditions::input_just_pressed};
use bevy_inspector_egui::prelude::*;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::prelude::Rotation;
use mimic::physics::PreviousTransform;
use mimic::rl::{is_rl_invalid, is_rl_valid, RLContext};
use mimic::urdf::{spawn_urdf, JointInfo, Robot, Root};
use pyo3::ffi::c_str;
use pyo3::Python;

#[derive(Debug, Resource, Reflect, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
enum URDFModel {
    Humanoid,
    Quadruped,
    Ant,
    Minitaur,
}

#[derive(Debug, Resource, Reflect, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct URDFModelConfig {
    model: URDFModel,
    root_pos: Vec3,
}

const HEALTHY_Y_RANGE: (f32, f32) = (0.2, 2.0);
const FORWARD_REWARD_WEIGHT: f32 = 1.0;
const CTRL_COST_WEIGHT: f32 = 0.5;
// const CONTACT_COST_WEIGHT: f32 = 0.0005;

fn main() {
    let mut app = App::new();

    // EXTREMELY IMPORTANT: Import torch on the main thread before interacting with pytorch anywhere else!!!!!!!!!!
    // https://github.com/PyO3/pyo3/issues/2611
    Python::with_gil(|py| Python::run(py, c_str!("import torch"), None, None).unwrap());

    app.insert_resource(URDFModelConfig {
        model: URDFModel::Ant,
        root_pos: Vec3::new(0., 1.5, 0.),
    })
    .register_type::<URDFModelConfig>()
    .insert_resource(RLContext::new(5000))
    .insert_resource(TimestepMode::Fixed {
        dt: 1.0 / 180.0,
        substeps: 3,
    })
    .add_plugins(DefaultPlugins)
    .add_plugins(WorldInspectorPlugin::new())
    .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
    .add_plugins(RapierDebugRenderPlugin::default())
    .add_plugins(PanOrbitCameraPlugin)
    .add_systems(Startup, setup_graphics)
    .add_systems(Startup, setup_physics)
    .add_systems(Startup, reset_model)
    .add_systems(
        Update,
        reset_model.run_if(input_just_pressed(KeyCode::KeyR)),
    )
    .add_systems(
        FixedFirst,
        (learn_model, reset_model.run_if(is_rl_invalid)).chain(),
    )
    .add_systems(
        FixedPreUpdate,
        observe_state
            .pipe(state_to_action)
            .pipe(apply_forces)
            .run_if(is_rl_valid),
    )
    .add_systems(
        FixedPostUpdate,
        (check_termination, update_previous_transforms, get_reward)
            .chain()
            .run_if(is_rl_valid),
    );

    #[cfg(debug_assertions)] // debug/dev builds only
    {
        use bevy::diagnostic::LogDiagnosticsPlugin;
        app.add_plugins(FrameTimeDiagnosticsPlugin)
            .add_plugins(LogDiagnosticsPlugin::default());
    }

    app.run();
}

fn setup_graphics(mut commands: Commands) {
    // Add a camera so we can see the debug-render.
    commands.spawn((
        PanOrbitCamera::default(),
        Transform::from_xyz(-3.0, 3.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn load_urdf_model(
    mut commands: Commands,
    urdf_model_config: Res<URDFModelConfig>,
    mut rl_context: ResMut<RLContext>,
) {
    let path = match urdf_model_config.model {
        URDFModel::Humanoid => "./assets/dmm-humanoid.urdf",
        URDFModel::Ant => "./assets/ant.urdf",
        URDFModel::Quadruped => "./assets/quadruped_2.urdf",
        URDFModel::Minitaur => "./assets/minitaur.urdf",
    };

    spawn_urdf(
        &mut commands,
        path,
        urdf_model_config.root_pos,
        &mut rl_context,
    );
    rl_context.valid = true;
}

fn setup_physics(
    mut commands: Commands,
    mut rapier_context: WriteDefaultRapierContext,
    mut time: ResMut<Time<Fixed>>,
) {
    /* Create the ground. */
    commands
        .spawn(Collider::cuboid(100.0, 0.1, 100.0))
        .insert(Transform::from_xyz(0.0, 0.0, 0.0));

    // due to the spring based solver model, joint constraints(DOFs) are imperfect
    // we can tune this parameter to make the joints somewhat stiffer
    rapier_context
        .integration_parameters
        .joint_natural_frequency = 5.0e6;

    // tune this parameter to accelerate the simulation
    // rapier_context.integration_parameters.dt
    time.set_timestep_hz(300.);
}

fn learn_model(mut rl_context: ResMut<RLContext>) {
    let timesteps = rl_context.elapsed_timestep;
    if timesteps >= rl_context.max_timestep {
        rl_context.learn().expect("Failed to learn model.");
        rl_context.elapsed_timestep = 0;
        rl_context.valid = false;
    }
}

// 1. Any of the state space values is no longer finite.
// 2. The y-coordinate of the torso (the height) is not in the closed interval given by the healthy_y_range argument (default is [0.2, 1.0]).
fn check_termination(
    mut rl_context: ResMut<RLContext>,
    query: Query<
        (&GlobalTransform, &Velocity, Entity, Option<&Root>),
        (With<Robot>, With<RigidBody>),
    >,
) {
    let should_end = query.iter().any(|(g_transform, velocity, _, is_root)| {
        if !g_transform.translation().is_finite()
            || !g_transform.rotation().is_finite()
            || !velocity.linvel.is_finite()
            || !velocity.angvel.is_finite()
        {
            return true;
        }
        if let Some(_) = is_root {
            let y = g_transform.translation().y;
            if y < HEALTHY_Y_RANGE.0 || y > HEALTHY_Y_RANGE.1 {
                return true;
            }
        }
        false
    });
    if should_end {
        tracing::info!("Environment terminated. Resetting environment.");
        // the reset will be triggered on the next update schedule
        rl_context.valid = false;
    }
}

fn reset_model(
    mut commands: Commands,
    urdf_model_config: Res<URDFModelConfig>,
    query: Query<Entity, With<RigidBody>>,
    mut rl_context: ResMut<RLContext>,
) {
    // clean up all entities
    rl_context.reset();
    for entity in query.iter() {
        commands.entity(entity).despawn_recursive();
    }
    load_urdf_model(commands, urdf_model_config, rl_context);
}

// IMPORTANT: this should run last after all processing is done
fn update_previous_transforms(
    mut query: Query<(&mut PreviousTransform, &GlobalTransform), With<Robot>>,
) {
    for (mut prev_transform, g_transform) in query.iter_mut() {
        prev_transform.0 = *g_transform;
    }
}

// reward = healthy_reward + forward_reward - ctrl_cost - contact_cost
fn get_reward(
    root_qry: Query<(&GlobalTransform, &PreviousTransform), (With<Robot>, With<Root>)>,
    time: Res<Time<Fixed>>,
    mut rl_context: ResMut<RLContext>,
) {
    let rl_context = &mut *rl_context;
    rl_context.elapsed_timestep += 1;

    // healthy reward - as long as ant is healthy(non terminated environment)
    let healthy_reward = 1.;

    // forward reward - weight * dx/dt
    let dt = time.delta_secs();
    let root = root_qry.get_single().unwrap();
    let dv = root.0.translation() - root.1 .0.translation();
    let dx = dv[0];
    let forward_reward = FORWARD_REWARD_WEIGHT * dx / dt;

    // control cost - weight * sum(u^2)
    let ctrl_cost = rl_context.ctrl_cost;
    rl_context.ctrl_cost = 0.;

    // TODO: contact cost - weight * sum(contact_force^2)
    // This is omitted for now as it is also semi-optional in gymnasium
    let total_reward = healthy_reward + forward_reward - ctrl_cost;
    rl_context
        .insert_reward(total_reward, !rl_context.valid)
        .expect("Failed to feed reward to model");
}

fn state_to_action(In(state): In<Vec<f32>>, mut rl_context: ResMut<RLContext>) -> Vec<f32> {
    let action = rl_context
        .get_action(state)
        .expect("Failed to get action from model.");

    // control cost - weight * sum(u^2)
    let ctrl_cost = action.iter().map(|u| u.powi(2)).sum::<f32>() * CTRL_COST_WEIGHT;
    rl_context.ctrl_cost = ctrl_cost;

    action
}

// qpos, qvel, cfrc_ext
fn observe_state(
    query: Query<
        (
            &GlobalTransform,
            &PreviousTransform,
            &Velocity,
            &ImpulseJoint,
            Entity,
        ),
        (With<Robot>, With<RigidBody>, Without<Root>),
    >,
    parent_query: Query<
        (&GlobalTransform, &PreviousTransform, Entity),
        (With<Robot>, With<RigidBody>),
    >,
    root_query: Query<
        (&GlobalTransform, &PreviousTransform, &Velocity, Entity),
        (With<Robot>, With<RigidBody>, With<Root>),
    >,
    time: Res<Time>,
) -> Vec<f32> {
    let mut states: Vec<f32> = Vec::new();

    // root-specific states
    let root = root_query.get_single().unwrap();
    let root_pos = root.0.translation();
    states.push(root_pos.x);
    states.push(root_pos.y);
    states.push(root_pos.z);
    let root_rot = root.0.rotation();
    states.push(root_rot.x);
    states.push(root_rot.y);
    states.push(root_rot.z);
    states.push(root_rot.w);

    let qpos = query
        .iter()
        .filter_map(|(g_transform, p_transform, velocity, joint, entity)| {
            if let TypedJoint::RevoluteJoint(revolute_joint) = joint.data {
                let child_rotation = g_transform.rotation();
                let child_rotation = Rotation::from(child_rotation);
                let parent_entity = joint.parent;
                let parent_rotation = parent_query
                    .iter()
                    .find(|(_, _, e)| e == &parent_entity)
                    .unwrap()
                    .0
                    .rotation();
                let parent_rotation = Rotation::from(parent_rotation);
                let angle = revolute_joint
                    .data
                    .raw
                    .as_revolute()
                    .unwrap()
                    .angle(&child_rotation, &parent_rotation);
                Some(angle)
            } else {
                None
            }
        })
        .collect::<Vec<f32>>();

    states.extend(qpos);

    // root-specific states
    let root_vel = root.2;
    states.push(root_vel.linvel.x);
    states.push(root_vel.linvel.y);
    states.push(root_vel.linvel.z);
    states.push(root_vel.angvel.x);
    states.push(root_vel.angvel.y);
    states.push(root_vel.angvel.z);

    let qvel = query
        .iter()
        .filter_map(|(g_transform, p_transform, _, joint, _)| {
            if let TypedJoint::RevoluteJoint(revolute_joint) = joint.data {
                let parent_entity = joint.parent;
                let parent_data = parent_query
                    .iter()
                    .find(|(_, _, e)| e == &parent_entity)
                    .unwrap();
                let parent_g_transform = parent_data.0;
                let parent_g_rotation = Rotation::from(parent_g_transform.rotation());
                let parent_p_transform = parent_data.1 .0;
                let parent_p_rotation = Rotation::from(parent_p_transform.rotation());

                let child_g_rotation = Rotation::from(g_transform.rotation());
                let child_p_rotation = Rotation::from(p_transform.0.rotation());

                let p_angle = revolute_joint
                    .data
                    .raw
                    .as_revolute()
                    .unwrap()
                    .angle(&child_p_rotation, &parent_p_rotation);
                let g_angle = revolute_joint
                    .data
                    .raw
                    .as_revolute()
                    .unwrap()
                    .angle(&child_g_rotation, &parent_g_rotation);

                Some((g_angle - p_angle) / time.delta_secs())
            } else {
                None
            }
        })
        .collect::<Vec<f32>>();

    states.extend(qvel);

    // TODO: handle cfrc_ext
    // let cfrc_ext = query
    //     .iter()
    //     .filter_map(|(_, _, _, force, _, _)| {
    //         Some(vec![
    //             force.force.x,
    //             force.force.y,
    //             force.force.z,
    //             force.torque.x,
    //             force.torque.y,
    //             force.torque.z,
    //         ])
    //     })
    //     .flatten()
    //     .collect::<Vec<f32>>();
    // states.extend(cfrc_ext);
    states
}

// apply forces to joints
fn apply_forces(
    In(forces): In<Vec<f32>>,
    mut impulse_joints: Query<(&mut ImpulseJoint, &JointInfo, Entity)>,
) {
    impulse_joints
        .iter_mut()
        .sort_by_key::<&JointInfo, _>(|x| x.index)
        .filter(|(_, b, _)| b.is_mobile)
        .for_each(|(mut joint, joint_info, _)| {
            let joint = &mut *joint;

            match joint.data {
                TypedJoint::RevoluteJoint(_) => {
                    let target_velocity = forces[joint_info.index];
                    // let target_velocity = target_velocity.clamp(-2., 2.);

                    joint
                        .data
                        .as_mut()
                        .set_motor_velocity(JointAxis::AngX, target_velocity, 1.);
                }
                TypedJoint::PrismaticJoint(_) => {
                    tracing::warn!("Prismatic joints are not supported yet.");
                }
                TypedJoint::FixedJoint(_) => {
                    tracing::error!("Fixed joints should not be applied forces.");
                }
                _ => {}
            }
        })
}
