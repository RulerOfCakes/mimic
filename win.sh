#!/bin/sh
# Use this script instead of `cargo run` to run the program natively on Windows in WSL2.
# You will need to have mingw-w64 installed.
cargo build --target x86_64-pc-windows-gnu &&
cp target/x86_64-pc-windows-gnu/debug/mimic.exe . &&
exec ./mimic.exe "$@"