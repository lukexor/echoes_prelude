//! Hot-reloadable library crate for `echoes_prelude`.

#![warn(
    anonymous_parameters,
    bare_trait_objects,
    clippy::branches_sharing_code,
    clippy::map_unwrap_or,
    clippy::match_wildcard_for_single_variants,
    // clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::needless_for_each,
    clippy::redundant_closure_for_method_calls,
    clippy::semicolon_if_nothing_returned,
    clippy::unreadable_literal,
    clippy::unwrap_used,
    clippy::expect_used,
    deprecated_in_future,
    ellipsis_inclusive_range_patterns,
    future_incompatible,
    missing_copy_implementations,
    missing_debug_implementations,
    // missing_docs,
    nonstandard_style,
    rust_2018_compatibility,
    rust_2018_idioms,
    rust_2021_compatibility,
    rustdoc::bare_urls,
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags,
    rustdoc::invalid_rust_codeblocks,
    rustdoc::private_intra_doc_links,
    single_use_lifetimes,
    trivial_casts,
    trivial_numeric_casts,
    unreachable_pub,
    unused,
    variant_size_differences
)]

pub mod logger;
#[macro_use]
pub mod profiling;
pub mod application;

mod event;
mod input;
mod platform;
mod renderer;

// pub use is required for `hot_lib_reloader`
pub use anyhow::Result;
pub use application::App;
pub use winit::window::Window;

#[no_mangle]
/// Initializes the logger correctly when the hot_reload is enabled.
pub fn app_init() {
    let _ = logger::init();
}

#[no_mangle]
pub fn update_and_render(app: &mut App, window: &Window) -> Result<()> {
    app.update_and_render(window)
}

#[no_mangle]
pub fn audio_samples(app: &mut App) -> Result<Vec<f32>> {
    app.audio_samples()
}
