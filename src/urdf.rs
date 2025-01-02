use crate::physics::PreviousTransform;
use crate::rl::{ModelArgs, RLContext};
use bevy::{
    prelude::*,
    utils::{hashbrown::HashMap, tracing},
};
use bevy_rapier3d::prelude::*;
use nalgebra::{Matrix3, Quaternion, SymmetricEigen, Unit, UnitQuaternion, Vector3};

#[derive(Component)]
pub struct Robot;

#[derive(Component)]
pub struct Root;

#[derive(Component)]
pub struct JointInfo {
    pub is_mobile: bool,
    pub index: usize,
    pub max_force: Option<f32>,
    pub max_velocity: Option<f32>, // TODO: enforce this constraint
    pub moment_arm: f32,
    pub damping: f32,
    pub friction: f32,
}

// spawns a URDF file as a Bevy entity.
// TODO: support mesh?
pub fn spawn_urdf(
    commands: &mut Commands,
    urdf_path: &str,
    root_pos: Vec3,
    rl_context: &mut RLContext,
) {
    let robot = xurdf::parse_urdf_from_file(urdf_path).unwrap_or_else(|e| {
        panic!("Failed to load URDF file: {}", e);
    });

    // used when setting up joints & modifying the link transforms.
    let mut link_entities: HashMap<String, Entity> = HashMap::new();
    let mut link_transforms: HashMap<String, Transform> = HashMap::new();
    // let mut link_col_transforms: HashMap<String, Transform> = HashMap::new();

    for link in robot.links {
        let mut link_entity = commands.spawn(RigidBody::Dynamic);
        link_entity.insert(Name::new(link.name.clone()));
        link_entity.insert(Velocity::default());

        // Assume 1 collider per link for now.
        // TODO: support multiple colliders per link.
        if link.collisions.len() > 1 {
            tracing::warn!(
                "More than 1 collision element found for link: {}",
                link.name
            );
        }

        if let Some(collision) = link.collisions.first() {
            let collider = match &collision.geometry {
                xurdf::Geometry::Box { size } => Collider::cuboid(
                    size[0] as f32 / 2.,
                    size[1] as f32 / 2.,
                    size[2] as f32 / 2.,
                ),
                xurdf::Geometry::Cylinder { radius, length } => {
                    Collider::cylinder((*length as f32) / 2f32, *radius as f32)
                }
                xurdf::Geometry::Capsule { radius, length } => {
                    Collider::capsule_z((*length as f32) / 2f32, *radius as f32)
                }
                xurdf::Geometry::Sphere { radius } => Collider::ball(*radius as f32),
                xurdf::Geometry::Mesh { .. } => {
                    todo!("Support mesh geometry for URDF collider.")
                }
            };

            // The collider's origin is the link's local frame, assuming the 1-collider-per-link convention.
            let mut self_transform = Transform {
                translation: Vec3::new(
                    collision.origin.xyz[0] as f32,
                    collision.origin.xyz[1] as f32,
                    collision.origin.xyz[2] as f32,
                ),
                rotation: Quat::from_rotation_z(collision.origin.rpy[2] as f32)
                    * Quat::from_rotation_y(collision.origin.rpy[1] as f32)
                    * Quat::from_rotation_x(collision.origin.rpy[0] as f32),
                scale: Vec3::ONE,
            };

            if let xurdf::Geometry::Cylinder { .. } = collision.geometry {
                let zup_to_yup =
                    Transform::from_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
                self_transform = self_transform * zup_to_yup;
            }
            // the collider's transform should be relative to the link's origin.
            link_entity.with_child((
                collider,
                self_transform,
                ColliderMassProperties::Density(0.),
            ));
            // enable CCD for all colliders.
            link_entity.insert(Ccd::enabled());
        }
        link_transforms.insert(link.name.clone(), Transform::IDENTITY);

        // mass properties
        {
            let mprops = mass_from_link_inertial(&link.inertial);
            link_entity.insert(AdditionalMassProperties::MassProperties(mprops));
            //   link_entity.insert(AdditionalMassProperties::Mass(0.001));
        }

        link_entities.insert(link.name.clone(), link_entity.id());
    }

    // In ideal cases(assumed), there should be 1 root link with indegree 0.
    let mut indegree: HashMap<String, usize> = HashMap::new();
    let mut adjacency_list: HashMap<String, Vec<String>> = HashMap::new();
    let mut mobile_joints: usize = 0;
    robot.joints.iter().for_each(|joint| {
        indegree
            .entry(joint.child.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);
        adjacency_list
            .entry(joint.parent.clone())
            .or_insert_with(Vec::new)
            .push(joint.child.clone());
        // the joint axis is in the joint's local frame = child link's local frame.
        let joint_axis = Vec3::new(
            joint.axis[0] as f32,
            joint.axis[1] as f32,
            joint.axis[2] as f32,
        );

        let joint_transform = Transform {
            translation: Vec3::new(
                joint.origin.xyz[0] as f32,
                joint.origin.xyz[1] as f32,
                joint.origin.xyz[2] as f32,
            ),
            rotation: Quat::from_rotation_z(joint.origin.rpy[2] as f32)
                * Quat::from_rotation_y(joint.origin.rpy[1] as f32)
                * Quat::from_rotation_x(joint.origin.rpy[0] as f32),
            scale: Vec3::ONE,
        };

        let anchor1 = joint_transform.translation;
        let parent_entity = link_entities
            .get(&joint.parent)
            .unwrap_or_else(|| panic!("Parent link not found: {}.", joint.parent));

        let child_entity = link_entities
            .get(&joint.child)
            .unwrap_or_else(|| panic!("Child link not found: {}.", joint.child));

        let child_transform = link_transforms
            .get(&joint.child)
            .unwrap_or_else(|| panic!("Parent link not found: {}.", joint.parent));
        let accumulated_transform = child_transform.mul_transform(joint_transform);
        link_transforms.insert(joint.child.clone(), accumulated_transform);

        let mut joint_info = JointInfo {
            is_mobile: true,
            index: 0,
            max_force: None,
            max_velocity: None,
            moment_arm: 0.,
            damping: 1., // default to 1 to avoid division by 0
            friction: 0.,
        };

        let joint: TypedJoint = match joint.joint_type.as_str() {
            "revolute" => {
                joint_info.max_force = Some(joint.limit.effort as f32);
                RevoluteJointBuilder::new(joint_axis)
                    .local_anchor1(anchor1)
                    .local_anchor2(Vec3::ZERO)
                    .limits([joint.limit.lower as f32, joint.limit.upper as f32]) // TODO: figure out a way to apply velocity limits.
                    .motor_max_force(joint.limit.effort as f32)
                    .motor_model(MotorModel::ForceBased)
                    .build()
                    .into()
            }
            "continuous" => {
                joint_info.max_force = Some(joint.limit.effort as f32);
                // continuous joints are just revolute joints without limits
                RevoluteJointBuilder::new(joint_axis)
                    .local_anchor1(anchor1)
                    .local_anchor2(Vec3::ZERO)
                    .motor_max_force(joint.limit.effort as f32)
                    .build()
                    .into()
            }
            "prismatic" => {
                todo!()
            }
            "fixed" => {
                joint_info.is_mobile = false;
                FixedJointBuilder::new()
                    .local_anchor1(anchor1)
                    .local_anchor2(Vec3::ZERO)
                    .build()
                    .into()
            }
            "floating" => {
                unimplemented!("Floating joints are not supported yet by rapier.");
            }
            "planar" => {
                unimplemented!("Planar joints are not supported yet by rapier.");
            }
            _ => {
                unimplemented!("Invalid joint type: {}", joint.joint_type);
            }
        };
        if joint_info.is_mobile {
            joint_info.index = mobile_joints;
            mobile_joints += 1;
        }

        let joint = ImpulseJoint::new(*parent_entity, joint);
        commands
            .entity(*child_entity)
            .insert(joint)
            .insert(joint_info);
        // TODO: insert names to joints as well?
    });

    let root = link_entities
        .iter()
        .find(|(name, _)| indegree.get(*name).is_none());
    let root_transform =
        Transform::from_rotation(Quat::from_axis_angle(Vec3::X, -std::f32::consts::FRAC_PI_2));
    let root_transform = Transform::from_translation(root_pos) * root_transform;

    // BFS over the adjacency list to set propagated transforms.

    let mut queue = Vec::new();
    if let Some((root_name, root_entity)) = root {
        link_transforms.insert(root_name.clone(), root_transform);
        queue.push(root_name.clone());
        commands.entity(*root_entity).insert(Root);
    }

    while !queue.is_empty() {
        let current = queue.remove(0);
        let current_transform: Transform = *link_transforms.get(&current).unwrap();
        let children = adjacency_list.get(&current);
        if let Some(children) = children {
            for child in children {
                let child_transform = link_transforms.get_mut(child).unwrap();
                *child_transform = current_transform.mul_transform(*child_transform);
                queue.push(child.clone());
            }
        }
    }
    link_transforms
        .iter()
        .for_each(|(link_name, link_transform)| {
            commands
                .entity(*link_entities.get(link_name).unwrap())
                .insert(*link_transform)
                .insert(PreviousTransform(GlobalTransform::from(*link_transform)))
                .insert(Robot);
        });

    // finally, initialize rl model
    rl_context
        .init_model(ModelArgs {
            obs_dim: 13 + mobile_joints * 2,
            act_dim: mobile_joints,
            ent_coeff: 0.00005,
            device: "cpu".to_string(),
            actor_lr: 0.0005,
            critic_lr: 0.0005,
            timesteps_per_batch: 7000,
            reward_scale: 0.1,
        })
        .expect("Failed to initialize RL model.");
}

// fn change_zup_to_yup(robot: xurdf::Robot) -> xurdf::Robot {}

fn compute_principal_inertia(
    inertia_matrix: Matrix3<f64>,
) -> (Vector3<f64>, Unit<Quaternion<f64>>) {
    // Perform eigen decomposition
    let eigen = SymmetricEigen::new(inertia_matrix);

    // Eigenvalues are the principal moments of inertia
    let principal_moments = eigen.eigenvalues;

    // Eigenvectors form the rotation matrix (principal axes)
    let rotation_matrix = eigen.eigenvectors;

    // Normalize the rotation matrix
    let rotation_matrix = rotation_matrix.normalize();

    // Convert the rotation matrix to a quaternion
    let rotation_quaternion = UnitQuaternion::from_matrix(&rotation_matrix);

    (principal_moments, rotation_quaternion)
}

fn mass_from_link_inertial(inertial: &xurdf::Inertial) -> MassProperties {
    let (principal_moments, rotation_quaternion) = compute_principal_inertia(inertial.inertia);
    let mass = if inertial.mass > 0. {
        inertial.mass as f32
    } else {
        0.0001 // assign negligible mass to avoid errors.
    };
    MassProperties {
        local_center_of_mass: Vec3::new(
            inertial.origin.xyz[0] as f32,
            inertial.origin.xyz[1] as f32,
            inertial.origin.xyz[2] as f32,
        ),
        mass,
        principal_inertia_local_frame: Quat::from_array([
            rotation_quaternion[0] as f32,
            rotation_quaternion[1] as f32,
            rotation_quaternion[2] as f32,
            rotation_quaternion[3] as f32,
        ]),
        // principal angular inertia
        principal_inertia: Vec3::new(
            principal_moments[0] as f32,
            principal_moments[1] as f32,
            principal_moments[2] as f32,
        ),
    }
}
