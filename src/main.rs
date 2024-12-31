use bevy::prelude::*;
use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, input::common_conditions::input_just_pressed};
use bevy_inspector_egui::prelude::*;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_rapier3d::prelude::*;
use mimic::urdf::spawn_urdf;

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

fn main() {
    let mut app = App::new();

    app.insert_resource(URDFModelConfig {
        model: URDFModel::Quadruped,
        root_pos: Vec3::new(0., 0., 0.),
    })
    .register_type::<URDFModelConfig>()
    .add_plugins(DefaultPlugins)
    .add_plugins(WorldInspectorPlugin::new())
    .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
    .add_plugins(RapierDebugRenderPlugin::default())
    .add_plugins(PanOrbitCameraPlugin)
    .add_systems(Startup, setup_graphics)
    .add_systems(Startup, setup_physics)
    .add_systems(Startup, load_urdfmodel)
    .add_systems(
        Update,
        reset_model.run_if(input_just_pressed(KeyCode::KeyR)),
    );

    #[cfg(debug_assertions)] // debug/dev builds only
    {
        // use bevy::diagnostic::LogDiagnosticsPlugin;
        app.add_plugins(FrameTimeDiagnosticsPlugin);
        // .add_plugins(LogDiagnosticsPlugin::default());
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

fn load_urdfmodel(mut commands: Commands, urdf_model_config: Res<URDFModelConfig>) {
    let path = match urdf_model_config.model {
        URDFModel::Humanoid => "./assets/dmm-humanoid.urdf",
        URDFModel::Ant => "./assets/ant.urdf",
        URDFModel::Quadruped => "./assets/quadruped_2.urdf",
        URDFModel::Minitaur => "./assets/minitaur.urdf",
    };
    // let from_mjcf = match urdf_model_config.model {
    //     URDFModel::Ant | URDFModel::Minitaur | URDFModel::Quadruped => true,
    //     _ => false,
    // };

    spawn_urdf(&mut commands, path, urdf_model_config.root_pos)
}

fn setup_physics(mut commands: Commands, mut rapier_context: WriteDefaultRapierContext) {
    /* Create the ground. */
    commands
        .spawn(Collider::cuboid(100.0, 0.1, 100.0))
        .insert(Transform::from_xyz(0.0, -2.0, 0.0));

    // due to the spring based solver model, joint constraints(DOFs) are imperfect
    // we can tune this parameter to make the joints somewhat stiffer
    rapier_context
        .integration_parameters
        .joint_natural_frequency = 5.0e6;
}

fn reset_model(
    mut commands: Commands,
    urdf_model_config: Res<URDFModelConfig>,
    query: Query<Entity, With<RigidBody>>,
) {
    // clean up all entities
    for entity in query.iter() {
        commands.entity(entity).despawn_recursive();
    }
    load_urdfmodel(commands, urdf_model_config);
}

fn query_forces(query: Query<&ExternalForce, With<RigidBody>>) {
    for force in query.iter() {
        println!("{:?}", force.force);
    }
}
