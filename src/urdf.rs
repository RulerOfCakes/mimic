use bevy::{prelude::*, utils::hashbrown::HashMap};
use bevy_rapier3d::prelude::*;
use nalgebra::{Matrix3, Quaternion, SymmetricEigen, Unit, UnitQuaternion, Vector3};
use rapier3d_urdf::{UrdfLoaderOptions, UrdfRobot};

#[derive(Component)]
pub struct Robot;

// spawns a URDF file as a Bevy entity.
pub fn spawn_urdf(commands: &mut Commands, urdf_path: &str) {
    let options = UrdfLoaderOptions {
        create_colliders_from_visual_shapes: true,
        create_colliders_from_collision_shapes: false,
        make_roots_fixed: true,
        // Z-up to Y-up.
        // shift: Isometry::rotation(Vector::x() * std::f32::consts::FRAC_PI_2),
        ..Default::default()
    };
    // TODO: support mesh?
    let (_, robot) = UrdfRobot::from_file(urdf_path, options, None).unwrap_or_else(|e| {
        panic!("Failed to load URDF file: {}", e);
    });

    // We cannot use this method without messing up bevy's graphics context.
    // humanoid.insert_using_impulse_joints(bodies, colliders, impulse_joints);
    // instead we will manually construct the entities.

    let root = commands.spawn(Robot).id();

    // used when setting up joints & modifying the transforms.
    let mut linkmap: HashMap<String, Entity> = HashMap::new();

    // Each link should have an associated rigid body.
    for link in robot.links {
        let mut link_entity = commands.spawn(RigidBody::Dynamic);
        link_entity.insert(Name::new(link.name.clone()));
        // Assume 1 collider per link for now.
        // TODO: support multiple colliders per link.
        if let Some(collision) = link.collisions.first() {
            let collider = match &collision.geometry {
                xurdf::Geometry::Box { size } => {
                    Collider::cuboid(size[0] as f32, size[1] as f32, size[2] as f32)
                }
                xurdf::Geometry::Cylinder { radius, length } => {
                    Collider::cylinder((*length as f32) / 2f32, *radius as f32)
                }
                xurdf::Geometry::Sphere { radius } => Collider::ball(*radius as f32),
                xurdf::Geometry::Mesh { filename, scale } => {
                    todo!("Support mesh geometry for URDF collider.")
                }
            };
            link_entity.insert(collider);
        }
        // mass properties
        {
            let mprops =
                ColliderMassProperties::MassProperties(mass_from_link_inertial(&link.inertial));
            link_entity.insert(mprops);
        }

        linkmap.insert(link.name, link_entity.id());
    }

    for joint in &robot.joints {
        let joint_axis = Vec3::new(
            joint.axis[0] as f32,
            joint.axis[1] as f32,
            joint.axis[2] as f32,
        );

        let joint = match joint.joint_type.as_str() {
            "revolute" => RevoluteJointBuilder::new(joint_axis).build(),
            "continuous" => {
                todo!()
            }
            "prismatic" => {
                todo!()
            }
            "fixed" => {
                todo!()
            }
            "floating" => {
                todo!()
            }
            "planar" => {
                todo!()
            }
            _ => {
                unimplemented!("Invalid joint type: {}", joint.joint_type);
            }
        };

        // let joint = ImpulseJoint::new(parent_entity, joint);
    }
}

fn compute_principal_inertia(
    inertia_matrix: Matrix3<f64>,
) -> (Vector3<f64>, Unit<Quaternion<f64>>) {
    // Perform eigen decomposition
    let eigen = SymmetricEigen::new(inertia_matrix);

    // Eigenvalues are the principal moments of inertia
    let principal_moments = eigen.eigenvalues;

    // Eigenvectors form the rotation matrix (principal axes)
    let rotation_matrix = eigen.eigenvectors;

    // Convert the rotation matrix to a quaternion
    let rotation_quaternion = UnitQuaternion::from_matrix(&rotation_matrix);

    (principal_moments, rotation_quaternion)
}

fn mass_from_link_inertial(inertial: &xurdf::Inertial) -> MassProperties {
    let (principal_moments, rotation_quaternion) = compute_principal_inertia(inertial.inertia);
    MassProperties {
        local_center_of_mass: Vec3::new(
            inertial.origin.xyz[0] as f32,
            inertial.origin.xyz[1] as f32,
            inertial.origin.xyz[2] as f32,
        ),
        mass: inertial.mass as f32,
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
