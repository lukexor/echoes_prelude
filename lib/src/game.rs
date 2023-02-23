//! Core game logic.

use crate::config::Config;
use anyhow::Result;
use pix_engine::prelude::*;

pub type Cx = Context<GameEvent, RenderContext>;

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
        std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;

        cx.set_projection(Mat4::identity().inverted_y());

        cx.load_mesh("viewport_mesh", "assets/meshes/rectangle.mesh");
        cx.load_texture("cyberpunk_test", "assets/textures/cyberpunk_test.tx");
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
    pub fn on_update(&mut self, cx: &mut Cx) -> Result<()> {
        // TODO: Reduce framerate when not focused.

        self.handle_input(cx)?;

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

    /// Render UI
    #[inline]
    pub fn render_imgui<'a>(
        &'a mut self,
        cx: &mut Cx,
        imgui: &'a mut imgui::ImGui,
    ) -> Result<&imgui::DrawData> {
        let ui = cx.new_ui_frame(imgui);
        if self.menu_is_open {
            ui.show_demo_window(&mut self.menu_is_open);
        } else {
        }
        cx.end_ui_frame(ui);
        Ok(cx.render_ui_frame(imgui))
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
