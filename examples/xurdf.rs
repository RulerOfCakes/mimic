fn main() {
    let robot = xurdf::parse_urdf_from_file("./assets/ant_torso.urdf");
    match robot {
        Ok(robot) => {
            println!("{:?}", robot.joints);
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
}
