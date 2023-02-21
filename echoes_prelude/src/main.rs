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
use pix_engine::{prelude::*, window::Positioned, Result as PixResult};

#[cfg(not(feature = "reload"))]
use echoes_prelude_lib::*;
#[cfg(feature = "reload")]
use hot_echoes_prelude_lib::*;

#[cfg(feature = "reload")]
#[hot_lib_reloader::hot_module(
    dylib = "echoes_prelude_lib",
    lib_dir = concat!(env!("CARGO_TARGET_DIR"), "/debug")
)]
mod hot_echoes_prelude_lib {
    pub(crate) use echoes_prelude_lib::*;
    hot_functions_from_file!("lib/src/lib.rs");

    // TODO: React to reloads to re-initialize if needed
    // See: https://github.com/rksm/hot-lib-reloader-rs/blob/master/examples/reload-events/src/main.rs
    // #[lib_change_subscription]
    // pub(crate) fn subscribe() -> hot_lib_reloader::LibReloadObserver {}
    #[lib_updated]
    pub(crate) fn was_updated() -> bool {}
}

const APPLICATION_NAME: &str = "Echoes: Prelude in Shadow";
const WINDOW_WIDTH: u32 = 1440;
const WINDOW_HEIGHT: u32 = 900;

fn main() -> Result<()> {
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
    imgui: imgui::ImGui,
    #[cfg(feature = "reload")]
    _trace_guard: TraceGuard,
}

impl Application {
    fn initialize() -> Result<Self> {
        let game = Game::new()?;
        Ok(Self {
            game,
            imgui: imgui::ImGui::create(),
            #[cfg(feature = "reload")]
            _trace_guard: initialize_trace(),
        })
    }
}

impl OnUpdate for Application {
    type UserEvent = GameEvent;
    type Renderer = RenderContext;

    /// Called on engine start.
    #[inline]
    fn on_start(&mut self, cx: &mut Cx) -> PixResult<()> {
        tracing::info!("echoes prelude start");
        self.game.initialize(cx)?;
        self.imgui.initialize(cx)?;
        Ok(())
    }

    /// Called every frame.
    #[inline]
    fn on_update(&mut self, cx: &mut Cx) -> PixResult<()> {
        #[cfg(feature = "reload")]
        if hot_echoes_prelude_lib::was_updated() {
            self._trace_guard = initialize_trace();
        }
        on_update(&mut self.game, cx)?;
        let _ = audio_samples(&mut self.game)?;
        Ok(())
    }

    /// Render UI.
    #[inline]
    fn render_imgui(&mut self, cx: &mut Cx) -> PixResult<&imgui::DrawData> {
        Ok(self.game.render_imgui(cx, &mut self.imgui)?)
    }

    /// Called on engine shutdown.
    #[inline]
    fn on_stop(&mut self, _cx: &mut Cx) {
        tracing::info!("echoes prelude stop");
    }

    /// Called on every event.
    #[inline]
    fn on_event(&mut self, cx: &mut Cx, event: Event<Self::UserEvent>) {
        self.imgui.on_event(event);
        on_event(&mut self.game, cx, event);
    }
}
