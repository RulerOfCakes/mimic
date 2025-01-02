use bevy::prelude::{Component, GlobalTransform, Quat};

#[derive(Component)]
pub struct PreviousTransform(pub GlobalTransform); // transform of last timestep, should be updated manually
pub fn calculate_angular_velocity(
    previous_rotation: Quat,
    current_rotation: Quat,
    dt: f32,
) -> Quat {
    let delta_rotation = current_rotation.conjugate() * previous_rotation;
    delta_rotation * (1. / dt)
}

// TODO: Fix
pub fn torque_to_motor_velocity(
    torque: f32,
    max_torque: Option<f32>,
    damping: f32,
    dt: f32,
) -> f32 {
    // Simple linear mapping with clamping:
    let desired_delta_velocity = if let Some(max_torque) = max_torque {
        torque.clamp(-max_torque, max_torque)
    } else {
        torque
    } * dt;
    desired_delta_velocity / damping
}
