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
use pix_engine::{config::FullscreenMode, prelude::*};

use echoes_prelude_lib::game::GameEvent;
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

const APPLICATION_NAME: &str = "Echoes: Prelude in Shadow";
const WINDOW_WIDTH: u32 = 1440;
const WINDOW_HEIGHT: u32 = 900;

const VERTEX_SHADER: &str = concat!(env!("OUT_DIR"), "/primary.vert.spv");
const FRAGMENT_SHADER: &str = concat!(env!("OUT_DIR"), "/primary.frag.spv");

#[tokio::main]
async fn main() -> pix_engine::Result<()> {
    logger::initialize()?;

    // TODO: tokio/reload https://github.com/rksm/hot-lib-reloader-rs/blob/master/examples/reload-events/src/main.rs
    #[cfg(feature = "hot_reload")]
    initialize_logger(); // Required to properly initialize logger with hot_reload

    let application = Application::initialize()?;
    let engine = Engine::builder()
        .title(APPLICATION_NAME)
        .version(env!("CARGO_PKG_VERSION"))
        .inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        // TODO: pull from saved configuration
        .fullscreen(false)
        .fullscreen_mode(FullscreenMode::Borderless)
        .cursor_grab(true)
        .shader(Shader::vertex("primary", VERTEX_SHADER).await?)
        .shader(Shader::fragment("primary", FRAGMENT_SHADER).await?)
        .build();
    engine.run(application)
}

#[derive(Debug)]
#[must_use]
struct Application {
    game: Game,
}

impl Application {
    fn initialize() -> Result<Self> {
        let game = Game::new()?;
        Ok(Self { game })
    }
}

impl Update for Application {
    type UserEvent = GameEvent;

    /// Called on engine start.
    fn on_start(&mut self, cx: &mut Context) -> pix_engine::Result<()> {
        log::info!("application started");
        self.game.initialize(cx);
        Ok(())
    }

    /// Called every frame.
    fn on_update(&mut self, delta_time: f32, cx: &mut Context) -> pix_engine::Result<()> {
        update(&mut self.game, delta_time, cx)?;
        render(&mut self.game, delta_time, cx)?;
        let _ = audio_samples(&mut self.game)?;
        Ok(())
    }

    /// Called on engine shutdown.
    fn on_stop(&mut self, _cx: &mut Context) {
        log::info!("application shutting down");
    }

    /// Called on every event.
    fn on_event(&mut self, delta_time: f32, event: Event<Self::UserEvent>, cx: &mut Context) {
        on_event(&mut self.game, delta_time, event, cx);
    }
}
