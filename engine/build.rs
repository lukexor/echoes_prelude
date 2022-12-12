use std::{
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

fn find_shaders(directory: &Path) -> io::Result<Vec<PathBuf>> {
    fs::read_dir(directory).map(|read_dir| {
        read_dir
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let path = entry.path();
                let extension = path.extension().and_then(OsStr::to_str);
                matches!(extension, Some("frag" | "vert")).then_some(path)
            })
            .collect::<Vec<PathBuf>>()
    })
}

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("valid OUT_DIR"));
    println!("cargo:rustc-env=OUT_DIR={}", out_dir.display());
    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/shaders");
    let shaders = find_shaders(&shader_dir)?;
    for shader in &shaders {
        println!("cargo:rerun-if-changed={}", shader.display());

        let filename = shader
            .file_name()
            .expect("valid shader filename")
            .to_string_lossy();
        let shader_out = out_dir.join(format!("{filename}.spv",));
        Command::new("glslc")
            .arg(shader)
            .arg("-o")
            .arg(shader_out)
            .status()?;
    }

    Ok(())
}
