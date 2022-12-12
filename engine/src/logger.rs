//! Logging methods.

use anyhow::Result;
use env_logger::{Builder, Env, Target};

/// Initialize the logging library.
pub fn initialize() -> Result<()> {
    // TODO: Add file logger
    Ok(Builder::from_env(Env::default().default_filter_or("info"))
        .target(Target::Stdout)
        .try_init()?)
}
