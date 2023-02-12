use anyhow::{bail, Result};
use asset_loader::{time, Asset, MeshAsset, TextureAsset};
use async_recursion::async_recursion;
use futures::future;
use std::{env, ffi::OsStr, path::PathBuf};
use tokio::task;
use tokio_stream::{wrappers::ReadDirStream, StreamExt};
use tracing_subscriber::EnvFilter;

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
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    let Some(directory) = env::args().nth(1) else {
        bail!("must provide an assets file path to convert");
    };

    tracing::info!("converting assets in {directory:?}");

    time!(total);
    future::join_all(
        find_assets(directory.into())
            .await?
            .into_iter()
            .filter_map(|filename| {
                filename
                    .extension()
                    .and_then(OsStr::to_str)
                    .and_then(|extension| {
                        Some(match extension {
                            "png" => TextureAsset::convert(filename.clone()),
                            "obj" => MeshAsset::convert(filename.clone()),
                            _ => return None,
                        })
                    })
                    .map(task::spawn)
            }),
    )
    .await;
    time!(end => total);

    tracing::info!("assets converted successfully");

    Ok(())
}
