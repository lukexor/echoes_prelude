use anyhow::{bail, Result};
use asset_loader::{time, Asset, MeshAsset, TextureAsset};
use async_recursion::async_recursion;
use std::{env, ffi::OsStr, path::PathBuf};
use tokio_stream::{wrappers::ReadDirStream, StreamExt};
use tracing::Level;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[async_recursion]
async fn find_assets(directory: PathBuf) -> Result<Vec<PathBuf>> {
    let mut assets = vec![];

    tracing::debug!("looking for assets in {directory:?}");

    if directory.is_dir() {
        let mut dirs = ReadDirStream::new(tokio::fs::read_dir(&directory).await?);
        while let Some(Ok(entry)) = dirs.next().await {
            let path = entry.path();
            if path.is_dir() {
                assets.append(&mut find_assets(path).await?);
            } else {
                let extension = path.extension().and_then(OsStr::to_str);
                if matches!(extension, Some("png" | "obj")) {
                    tracing::debug!("found asset {path:?}");
                    assets.push(path);
                }
            }
        }
    }

    Ok(assets)
}

#[tokio::main]
async fn main() -> Result<()> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();
    let registry = tracing_subscriber::registry().with(env_filter);
    let registry = registry.with(
        fmt::Layer::new()
            .compact()
            .without_time()
            .with_line_number(true)
            .with_writer(std::io::stderr),
    );
    registry.try_init()?;

    let Some(directory) = env::args().nth(1) else {
        bail!("must provide an assets file path to convert");
    };

    tracing::info!("converting assets in {directory:?}");

    time!(total);
    let asset_files = find_assets(directory.into()).await?;
    for filename in asset_files {
        match filename.extension().and_then(OsStr::to_str) {
            Some("png") => {
                TextureAsset::convert(filename).await?;
            }
            Some("obj") => {
                MeshAsset::convert(filename).await?;
            }
            _ => (),
        }
    }
    time!(end => total);

    tracing::info!("assets converted successfully");

    Ok(())
}
