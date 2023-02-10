use anyhow::{Context, Result};
use bytes::Bytes;
use derive_more::{From, TryInto};
use std::path::{Path, PathBuf};
use std::{fs::File as FileSync, io::BufReader as BufReaderSync};
use tokio::fs::create_dir_all;
use tokio::{
    fs::File,
    io::{BufReader, BufWriter},
};

#[derive(Debug, Clone, PartialEq, From, TryInto)]
#[must_use]
pub enum DataSource {
    #[from(forward)]
    Path(PathBuf),
    Bytes(Bytes),
    // TODO: Could add Network(Ipv4Addr) or Database
}

#[inline]
pub async fn create_file(path: impl AsRef<Path>) -> Result<BufWriter<File>> {
    let path = path.as_ref();
    let directory = path.parent().context("can not create file at root path")?;
    create_dirs(directory).await?;
    Ok(BufWriter::new(File::create(path).await.with_context(
        || format!("failed to create file for writing: {path:?}"),
    )?))
}

#[inline]
pub async fn open_file(path: impl AsRef<Path>) -> Result<BufReader<File>> {
    let path = path.as_ref();
    Ok(BufReader::new(File::open(path).await.with_context(
        || format!("failed to open file for reading: {path:?}"),
    )?))
}

#[inline]
pub fn open_file_sync(path: impl AsRef<Path>) -> Result<BufReaderSync<FileSync>> {
    let path = path.as_ref();
    Ok(BufReaderSync::new(FileSync::open(path).with_context(
        || format!("failed to open file for reading: {path:?}"),
    )?))
}

#[inline]
pub async fn create_dirs(directory: impl AsRef<Path>) -> Result<()> {
    let directory = directory.as_ref();
    if !directory.exists() {
        create_dir_all(directory)
            .await
            .with_context(|| format!("failed to create directory {directory:?}"))?;
    }
    Ok(())
}
