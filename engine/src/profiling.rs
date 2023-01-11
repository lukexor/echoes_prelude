//! Profiling and debugging methods.

/// Start a profiling timer for a section of code.
#[macro_export]
macro_rules! time {
    ($label:ident) => {
        use ::std::time::Instant;
        let mut $label = Some(Instant::now());
    };
}

/// Log the current running profling timer.
#[macro_export]
macro_rules! timeLog {
    ($label:ident) => {
        match $label {
            Some(label) => {
                tracing::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32())
            }
            None => tracing::warn!("Timer `{}` has been terminated.", stringify!($label)),
        };
    };
}

/// End the profiling timer.
#[macro_export]
macro_rules! timeEnd {
    ($label:ident) => {{
        match $label.take() {
            Some(label) => {
                tracing::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32())
            }
            None => tracing::warn!("Timer `{}` has been terminated.", stringify!($label)),
        };
    }};
}
