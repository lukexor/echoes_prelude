use anyhow::{Context, Result};
use std::{
    env,
    ffi::OsStr,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
};
use tokio::fs;
use tokio_stream::{wrappers::ReadDirStream, StreamExt};

async fn find_shaders(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut shaders = vec![];

    let mut dirs = ReadDirStream::new(
        fs::read_dir(&directory)
            .await
            .with_context(|| format!("failed to read directory: {directory:?}"))?,
    );
    while let Some(Ok(entry)) = dirs.next().await {
        let path = entry.path();
        let extension = path.extension().and_then(OsStr::to_str);
        if matches!(extension, Some("frag" | "vert")) {
            shaders.push(path);
        }
    }

    Ok(shaders)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("valid OUT_DIR"));
    println!("cargo:rustc-env=OUT_DIR={}/", out_dir.display());

    let assets_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    asset_loader::convert_all(&assets_dir, &out_dir).await?;

    let shader_dir = assets_dir.join("shaders");
    let shaders = find_shaders(&shader_dir)
        .await
        .with_context(|| format!("failed to find shaders in {shader_dir:?}"))?;
    for shader in &shaders {
        println!("cargo:rerun-if-changed={}", shader.display());

        let filename = shader
            .file_name()
            .expect("valid shader filename")
            .to_string_lossy();
        let shader_out = out_dir.join(format!("{filename}.spv"));
        // TODO: Check for glslc being installed
        let output = Command::new("glslc")
            .arg(shader)
            .arg("-o")
            .arg(&shader_out)
            .output()
            .with_context(|| format!("failed to compile shader: {shader:?} -> {shader_out:?}"))?;
        if !output.status.success() {
            io::stdout()
                .write_all(&output.stdout)
                .expect("write to stdout");
            io::stderr()
                .write_all(&output.stderr)
                .expect("write to stderr");
            panic!("failed to compile shaders");
        }
    }

    Ok(())
}
