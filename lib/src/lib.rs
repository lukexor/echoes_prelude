//! Hot-reloadable library for `echoes_prelude`.

#![warn(
    anonymous_parameters,
    bare_trait_objects,
    clippy::branches_sharing_code,
    clippy::map_unwrap_or,
    clippy::match_wildcard_for_single_variants,
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
    // TODO: Re-enable
    // clippy::missing_errors_doc,
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

pub mod config;
pub mod game;
pub mod trace;

// NOTE: pub use is required for any types used in hot-reloadable functions to expose them to
// the `hot_lib_reloader` macro.
pub use anyhow::Result;
pub use game::{Cx, Game, GameEvent};
pub use pix_engine::imgui::Ui;
pub use pix_engine::prelude::*;
pub use trace::TraceGuard;

/// Hot-reloadable [trace::initialize]. Because globals are not shared across dynamic library
/// boundaries, tracing has to be re-initialized on every reload.
#[inline]
#[no_mangle]
pub fn initialize_trace() -> TraceGuard {
    trace::initialize()
}

/// Hot-reloadable [Game::on_update].
#[inline]
#[no_mangle]
pub fn on_update(game: &mut Game, cx: &mut Cx, ui: &mut Ui) -> Result<()> {
    game.on_update(cx, ui)
}

/// Hot-reloadable [Game::on_reload].
#[inline]
#[no_mangle]
pub fn on_reload(game: &mut Game, cx: &mut Cx) -> Result<()> {
    game.on_reload(cx)
}

/// Hot-reloadable [Game::audio_samples].
#[inline]
#[no_mangle]
pub fn audio_samples(game: &mut Game) -> Result<Vec<f32>> {
    game.audio_samples()
}

/// Hot-reloadable [Game::on_event].
#[inline]
#[no_mangle]
pub fn on_event(game: &mut Game, cx: &mut Cx, event: Event<GameEvent>) {
    game.on_event(cx, event);
}
