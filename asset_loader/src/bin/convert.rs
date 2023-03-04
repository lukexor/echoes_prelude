use anyhow::{bail, Result};
use std::env;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    match env::args().nth(1) {
        Some(directory) => asset_loader::convert_all(directory).await?,
        None => bail!("must provide an assets file path to convert"),
    }

    Ok(())
}
