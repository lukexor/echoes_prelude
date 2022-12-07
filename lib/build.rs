use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RENDERER");
    match env::var("RENDERER")
        .unwrap_or_else(|_| "vulkan".into())
        .to_lowercase()
        .as_ref()
    {
        "vulkan" => println!("cargo:rustc-cfg=vulkan"),
        "opengl" => println!("cargo:rustc-cfg=opengl"),
        renderer => panic!("`{renderer}` is not a supported renderer backend"),
    }
}
