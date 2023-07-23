//! Core game logic.

use crate::config::Config;
use anyhow::Result;
use pix_engine::{imgui::Ui, mesh::Mesh, prelude::*};

pub type Cx = Context<GameEvent, RenderContext>;

const NEAR_CLIP: f32 = 0.001;
const FAR_CLIP: f32 = 1000.0;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GameEvent {
    Debug, // FIXME: Temporary
}

#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct Game {
    config: Config,
    menu_is_open: bool,
}

impl Game {
    /// Create the `Game` instance.
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: Config::new(),
            menu_is_open: false,
        })
    }

    /// Initialize the `Game` instance.
    #[inline]
    pub fn initialize(&mut self, cx: &mut Cx) -> Result<()> {
        tracing::debug!("initializing echoes prelude");

        let ratio = cx.height() as f32 / cx.width() as f32;
        cx.set_projection(
            Mat4::orthographic(-1.0, 1.0, -1.0 * ratio, 1.0 * ratio, NEAR_CLIP, FAR_CLIP)
                .inverted_y(),
        );

        cx.load_mesh("viewport_mesh", Mesh::RECTANGLE);
        cx.load_texture("cyberpunk_test", "lib/assets/textures/cyberpunk_test.tx");
        cx.load_object(
            "viewport",
            "viewport_mesh",
            MaterialType::Texture("cyberpunk_test".into()),
            Mat4::identity(),
        );

        tracing::debug!("initialized echoes prelude successfully");

        Ok(())
    }

    /// Hot-reload global state.
    #[inline]
    pub fn on_reload(&mut self, _cx: &mut Cx) -> Result<()> {
        Ok(())
    }

    /// Called every frame to update game state.
    #[inline]
    pub fn on_update(&mut self, cx: &mut Cx, ui: &mut Ui) -> Result<()> {
        // TODO: Reduce framerate when not focused.

        self.handle_input(cx)?;

        if self.menu_is_open {
            ui.show_demo_window(&mut self.menu_is_open);
        } else {
        }

        Ok(())
    }

    #[inline]
    fn handle_input(&mut self, cx: &mut Cx) -> Result<()> {
        // FIXME: temporary
        if cx.key_typed(KeyCode::T) {
            cx.send_event(GameEvent::Debug);
        }

        #[cfg(debug_assertions)]
        if cx.key_typed(KeyCode::Escape) {
            cx.quit();
        }

        if cx.key_typed(KeyCode::M) {
            self.menu_is_open = !self.menu_is_open;
        }

        if cx.key_typed(KeyCode::Return) && cx.modifiers_down(ModifierKeys::CTRL) {
            cx.toggle_fullscreen(self.config.fullscreen_mode);
        }

        Ok(())
    }

    /// Called every frame to retrieve audio samples to be played.
    #[inline]
    pub fn audio_samples(&mut self) -> Result<Vec<f32>> {
        // TODO audio_samples https://github.com/RustAudio/cpal?
        Ok(vec![])
    }

    /// Called on every event.
    #[inline]
    pub fn on_event(&mut self, cx: &mut Cx, event: Event<GameEvent>) {
        if !cx.focused() {
            return;
        }

        if let Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::CloseRequested | WindowEvent::Destroyed { .. } => cx.quit(),
                _ => (),
            }
        }
    }
}
