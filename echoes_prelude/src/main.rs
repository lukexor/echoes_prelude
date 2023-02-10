#![doc = include_str!("../../README.md")]
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
use pix_engine::{prelude::*, window::Positioned};

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

    // TODO: React to reloads to re-initialize if needed
    #[lib_change_subscription]
    pub(crate) fn subscribe() -> hot_lib_reloader::LibReloadObserver {}
}

const APPLICATION_NAME: &str = "Echoes: Prelude in Shadow";
const WINDOW_WIDTH: u32 = 1440;
const WINDOW_HEIGHT: u32 = 900;

fn main() -> Result<()> {
    // TODO: tokio/reload https://github.com/rksm/hot-lib-reloader-rs/blob/master/examples/reload-events/src/main.rs
    // TODO: Re-init logger on hot reload
    let _trace = trace::initialize();
    run_application()
}

fn run_application() -> Result<()> {
    let application = Application::initialize()?;
    let engine = Engine::builder()
        .title(APPLICATION_NAME)
        .version(env!("CARGO_PKG_VERSION"))
        .inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .positioned(Positioned::Center)
        // TODO: pull from saved configuration
        // .fullscreen(Fullscreen::Borderless)
        .cursor_grab(true)
        .build();
    engine.run(application)?;
    Ok(())
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

impl OnUpdate for Application {
    type UserEvent = GameEvent;

    /// Called on engine start.
    fn on_start(&mut self, cx: &mut Context<'_, Self::UserEvent>) -> pix_engine::Result<()> {
        tracing::info!("application started");
        self.game.initialize(cx)?;
        Ok(())
    }

    /// Called every frame.
    fn on_update(&mut self, cx: &mut Context<'_, Self::UserEvent>) -> pix_engine::Result<()> {
        on_update(&mut self.game, cx)?;
        let _ = audio_samples(&mut self.game)?;
        Ok(())
    }

    /// Called on engine shutdown.
    fn on_stop(&mut self, _cx: &mut Context<'_, Self::UserEvent>) {
        tracing::info!("application shutting down");
    }

    /// Called on every event.
    fn on_event(&mut self, cx: &mut Context<'_, Self::UserEvent>, event: Event<Self::UserEvent>) {
        on_event(&mut self.game, cx, event);
    }
}
