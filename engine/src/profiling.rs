//! Profiling and debugging methods.

/// Create a profiling timer for a section of code.
#[macro_export]
macro_rules! time {
    ($label:ident) => {
        let mut $label = Some(::std::time::Instant::now());
    };
    (log => $label:ident) => {
        match $label {
            Some(label) => {
                ::tracing::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32())
            }
            None => tracing::warn!("Timer `{}` has been terminated.", stringify!($label)),
        };
    };
    (end => $label:ident) => {{
        match $label.take() {
            Some(label) => {
                ::tracing::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32())
            }
            None => ::tracing::warn!("Timer `{}` has been terminated.", stringify!($label)),
        };
    }};
}
