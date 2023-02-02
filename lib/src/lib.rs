//! Hot-reloadable library for `echoes_prelude`.

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

pub mod config;
pub mod game;
pub mod trace;

// pub use is required all exposed types for `hot_lib_reloader`
pub use anyhow::Result;
pub use game::{Game, GameEvent};
pub use pix_engine::prelude::*;

#[no_mangle]
pub fn on_update(game: &mut Game, cx: &mut Context<'_, GameEvent>) -> Result<()> {
    game.on_update(cx)
}

#[no_mangle]
pub fn audio_samples(game: &mut Game) -> Result<Vec<f32>> {
    game.audio_samples()
}

#[no_mangle]
pub fn on_event(game: &mut Game, cx: &mut Context<'_, GameEvent>, event: Event<GameEvent>) {
    game.on_event(cx, event);
}
