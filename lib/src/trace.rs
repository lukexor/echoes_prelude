//! Logging methods.
//! .finish()

use tracing::Level;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

const LOG_DIR: &str = "logs";
const LOG_FILENAME_PREFIX: &str = "echoes_prelude.log";

#[derive(Debug)]
#[must_use]
pub struct TraceGuard {
    _file_log_guard: WorkerGuard,
}

/// Initialize the tracing library.
pub fn initialize() -> TraceGuard {
    let env_filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();

    let registry = tracing_subscriber::registry().with(env_filter);
    // TODO: Add max_log_files when https://github.com/tokio-rs/tracing/pull/2323 is released
    let file_appender = tracing_appender::rolling::daily(LOG_DIR, LOG_FILENAME_PREFIX);
    let (non_blocking_file, _file_log_guard) = tracing_appender::non_blocking(file_appender);
    let registry = registry.with(
        fmt::Layer::new()
            .compact()
            .with_line_number(true)
            .with_writer(non_blocking_file),
    );

    #[cfg(debug_assertions)]
    let registry = registry.with(
        fmt::Layer::new()
            .compact()
            .without_time()
            .with_line_number(true)
            .with_writer(std::io::stderr),
    );

    if let Err(err) = registry.try_init() {
        eprintln!("setting tracing default failed: {err:?}");
    }

    TraceGuard { _file_log_guard }
}
