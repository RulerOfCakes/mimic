use bevy::prelude::*;
use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, input::common_conditions::input_just_pressed};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_rapier3d::prelude::*;
use mimic::urdf::spawn_urdf;

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins)
        .add_plugins(WorldInspectorPlugin::new())
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup_graphics)
        .add_systems(Startup, setup_physics)
        .add_systems(Startup, load_humanoid)
        .add_systems(
            Update,
            reset_humanoid.run_if(input_just_pressed(KeyCode::KeyR)),
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

fn load_humanoid(mut commands: Commands) {
    spawn_urdf(
        &mut commands,
        "./assets/dmm-humanoid.urdf",
        Vec3::new(0., 5., 0.),
    );
}

fn setup_physics(
    mut commands: Commands,
    mut rapier_config: Query<&mut RapierConfiguration, With<RapierContext>>,
) {
    /* Create the ground. */
    commands
        .spawn(Collider::cuboid(100.0, 0.1, 100.0))
        .insert(Transform::from_xyz(0.0, -2.0, 0.0));

    /* Create the bouncing ball. */
    commands
        .spawn(RigidBody::Dynamic)
        .insert(Collider::ball(0.5))
        .insert(Restitution::coefficient(0.7))
        .insert(Transform::from_xyz(0.0, 4.0, 0.0));
}

fn reset_humanoid(mut commands: Commands, query: Query<Entity, With<RigidBody>>) {
    // clean up all entities
    for entity in query.iter() {
        commands.entity(entity).despawn_recursive();
    }
    load_humanoid(commands);
}
