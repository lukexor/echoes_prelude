#![doc = include_str!("../README.md")]
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

use anyhow::Result;
use pix_engine::prelude::*;

#[cfg(not(feature = "hot_reload"))]
use echoes_prelude_lib::*;
#[cfg(feature = "hot_reload")]
use hot_echoes_prelude_lib::*;

#[cfg(feature = "hot_reload")]
#[hot_lib_reloader::hot_module(
    dylib = "echoes_prelude_lib",
    lib_dir = concat!(env!("CARGO_TARGET_DIR"), "/debug")
)]
mod hot_echoes_prelude_lib {
    pub(crate) use echoes_prelude_lib::*;
    hot_functions_from_file!("engine/src/lib.rs");

    #[lib_change_subscription]
    pub(crate) fn _subscribe() -> hot_lib_reloader::LibReloadObserver {}
}

const APPLICATION_NAME: &str = "Echoes: Prelude";
// TODO: fullscreen?
const WINDOW_WIDTH: u32 = 1440;
const WINDOW_HEIGHT: u32 = 900;

const VERTEX_SHADER: &str = concat!(env!("OUT_DIR"), "/primary.vert.spv");
const FRAGMENT_SHADER: &str = concat!(env!("OUT_DIR"), "/primary.frag.spv");

// TODO: async io
fn main() -> pix_engine::Result<()> {
    logger::initialize()?;

    // TODO: tokio/reload https://github.com/rksm/hot-lib-reloader-rs/blob/master/examples/reload-events/src/main.rs
    #[cfg(feature = "hot_reload")]
    initialize_logger(); // Required to properly initialize logger with hot_reload

    let application = Application::initialize()?;
    let engine = Engine::builder()
        .title(APPLICATION_NAME)
        .dimensions(WINDOW_WIDTH, WINDOW_HEIGHT)
        .shader(Shader::vertex("primary", VERTEX_SHADER)?)
        .shader(Shader::fragment("primary", FRAGMENT_SHADER)?)
        .build()?;
    Engine::run(engine, application)
}

#[derive(Debug)]
#[must_use]
pub struct Application {
    game: Game,
}

impl Application {
    pub fn initialize() -> Result<Self> {
        let game = Game::initialize()?;
        Ok(Self { game })
    }
}

impl Update for Application {
    fn on_start(&mut self, _cx: &mut Context) -> pix_engine::Result<()> {
        log::info!("application started");
        Ok(())
    }

    fn on_update(&mut self, delta_time: f32, cx: &mut Context) -> pix_engine::Result<()> {
        update(&mut self.game, delta_time, cx)?;
        render(&mut self.game, delta_time, cx)?;
        let _ = audio_samples(&mut self.game)?;
        Ok(())
    }

    fn on_stop(&mut self, _cx: &mut Context) {
        log::info!("application shutting down");
    }

    fn on_event(&mut self, event: Event, cx: &mut Context) {
        on_event(&mut self.game, event, cx);
    }
}
