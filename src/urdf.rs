use bevy::{
    prelude::*,
    utils::{hashbrown::HashMap, tracing},
};
use bevy_rapier3d::prelude::*;
use nalgebra::{Matrix3, Quaternion, SymmetricEigen, Unit, UnitQuaternion, Vector3};
use rapier3d_urdf::{UrdfLoaderOptions, UrdfRobot};

#[derive(Component)]
pub struct Robot;

// spawns a URDF file as a Bevy entity.
pub fn spawn_urdf(commands: &mut Commands, urdf_path: &str, root_pos: Vec3) {
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
    // humanoid.insert_using_impulse_joints(bodies, colliders, impulse_joints)

    // used when setting up joints & modifying the link transforms.
    let mut link_entities: HashMap<String, Entity> = HashMap::new();
    let mut link_transforms: HashMap<String, Transform> = HashMap::new();
    // let mut link_col_transforms: HashMap<String, Transform> = HashMap::new();

    for link in robot.links {
        let mut link_entity = commands.spawn(RigidBody::Dynamic);
        link_entity.insert(Name::new(link.name.clone()));

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

            // link_col_transforms.insert(link.name.clone(), self_transform);

            // the collider's transform should be relative to the link's origin.
            link_entity.with_child((collider, self_transform));
        } else {
            // link_col_transforms.insert(link.name.clone(), Transform::IDENTITY);
        }
        link_transforms.insert(link.name.clone(), Transform::IDENTITY);

        // mass properties
        {
            let mprops =
                ColliderMassProperties::MassProperties(mass_from_link_inertial(&link.inertial));
            link_entity.insert(mprops);
        }

        link_entities.insert(link.name.clone(), link_entity.id());
    }

    // In ideal cases(assumed), there should be 1 root link with indegree 0.
    let mut indegree: HashMap<String, usize> = HashMap::new();
    // required to easily compute the transform chain
    let mut adjacency_list: HashMap<String, Vec<String>> = HashMap::new();

    for joint in &robot.joints {
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

        let joint: TypedJoint = match joint.joint_type.as_str() {
            "revolute" => RevoluteJointBuilder::new(joint_axis)
                .local_anchor1(anchor1)
                .local_anchor2(Vec3::ZERO)
                .limits([joint.limit.lower as f32, joint.limit.upper as f32]) // TODO: figure out a way to apply effort & velocity limits.
                .motor_max_force(joint.limit.effort as f32)
                .build()
                .into(),
            "continuous" => {
                unimplemented!("Continuous joints are not supported yet by rapier.");
            }
            "prismatic" => {
                todo!()
            }
            "fixed" => FixedJointBuilder::new()
                .local_anchor1(anchor1)
                .local_anchor2(Vec3::ZERO)
                .build()
                .into(),
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
        let joint = ImpulseJoint::new(*parent_entity, joint);
        commands.entity(*child_entity).insert(joint);
        // TODO: insert names to joints as well?
    }

    let root = link_entities
        .iter()
        .find(|(name, _)| indegree.get(*name).is_none());

    // BFS over the adjacency list to set propagated transforms.

    let mut queue = Vec::new();
    if let Some((root_name, _)) = root {
        queue.push(root_name.clone());
    }

    while !queue.is_empty() {
        let current = queue.remove(0);
        let current_transform = *link_transforms.get(&current).unwrap();
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
                .insert(Robot);
        });
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
