# Troubleshooting

These are some issues that you may encounter while working with this project.

1. invalid argument for exe
    - It seems to randomly occur with `bevy` / `bevy_rapier3d` whenever you cross-compile the project with certain
      environment variables, like `RUST_BACKTRACE=1`. The solution is to remove the environment variable from your
      config file(`.cargo/config.toml`) and recompile the project, however the root cause is still uncertain.
    - Another cause seems to be the usage of the dynamic linking feature on bevy. However since this feature is actually
      recommended by Bevy to reduce build times, I suppose this only happens when you cross-compile the project(e.g.
      from linux to windows).
    - If you are working on WSL, it is highly recommended to work on windows natively. Despite several positive claims,
      my experience is that bevy on WSL sometimes display unstable behavior(like this issue) with seemingly no
      correlation
      to code.
2. parent-child hierarchy of multiple rigidbodies
    - Bevy's parent-child local transform hierarchy is not smoothly supported by `bevy_rapier3d`. You should avoid bevy
      hierachies when dealing with rapier rigidbodies if you experience any sort of unexpected issue(misbehaving
      transforms) with this setup.
3. invalid/misbehaving URDFs
    - Ensure that the URDF itself is valid by using external visualization tools, such as
      this [web viewer](http://urdf.robotsfan.com/).
    - URDFs converted from other formats like MJCF will most likely be invalid. I have experimented with many converter
      tools and many proved to be fallible.
4. Pytorch operations randomly failing with invalid memory access
    - Ensure that you imported the pytorch module on the main thread first before anywhere else. This is a bug related
      to pytorch and pyo3.
    - https://github.com/PyO3/pyo3/issues/2611