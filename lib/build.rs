use anyhow::Result;
use std::{env, path::PathBuf};

#[tokio::main]
async fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("valid OUT_DIR"));
    println!("cargo:rustc-env=OUT_DIR={}", out_dir.display());

    let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    asset_loader::convert_all(asset_dir).await?;

    Ok(())
}
