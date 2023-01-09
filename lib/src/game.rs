//! Core game logic.

use crate::config::Config;
use anyhow::Result;
use pix_engine::{camera::Camera, prelude::*};

pub type GameEvent = ();

#[allow(missing_copy_implementations)]
#[derive(Debug, Clone)]
#[must_use]
pub struct Game {
    config: Config,
    camera: Camera,
    projection: Mat4,
    is_dirty: bool,
    model: Mat4, // FIXME: temporary
    near_clip: f32,
    far_clip: f32,
}

impl Game {
    /// Create the `Game` instance.
    pub fn new() -> Result<Self> {
        let near_clip = 0.1;
        let far_clip = 100.0;
        Ok(Self {
            config: Config::new(),
            camera: Camera::new(vector!(0.0, 0.5, 3.0)),
            projection: Mat4::identity(),
            is_dirty: true,
            model: Mat4::identity(), // FIXME: temporary
            near_clip,
            far_clip,
        })
    }

    /// Initialize the `Game` instance.
    pub fn initialize(&mut self, cx: &mut Context) {
        self.update_projection(cx);
    }

    /// Called every frame to update game state.
    pub fn update(&mut self, delta_time: f32, cx: &mut Context) -> Result<()> {
        // TODO: Reduce framerate when not focused.

        self.handle_events(delta_time, cx);
        self.update_projection(cx);
        let time = cx.elapsed().as_secs_f32();
        // FIXME: temporary
        self.model = Mat4::rotation(
            Radians(-90f32.to_radians()),
            Radians(0.0),
            Radians(-120f32.to_radians() + time * 20f32.to_radians()),
        );
        Ok(())
    }

    fn handle_events(&mut self, delta_time: f32, cx: &mut Context) {
        let speed = 5.0 * delta_time;
        if cx.key_down(KeyCode::A) {
            self.camera.move_left(speed);
        }
        if cx.key_down(KeyCode::D) {
            self.camera.move_right(speed);
        }
        if cx.key_down(KeyCode::W) {
            self.camera.move_forward(speed);
        }
        if cx.key_down(KeyCode::S) {
            self.camera.move_backward(speed);
        }
        if cx.key_down(KeyCode::Left) {
            self.camera.yaw(Degrees(-20.0 * speed));
        }
        if cx.key_down(KeyCode::Right) {
            self.camera.yaw(Degrees(20.0 * speed));
        }
        if cx.key_down(KeyCode::Up) {
            self.camera.pitch(Degrees(-20.0 * speed));
        }
        if cx.key_down(KeyCode::Down) {
            self.camera.pitch(Degrees(20.0 * speed));
        }
        if cx.key_down(KeyCode::Space) {
            self.camera.move_up(speed);
        }
        if cx.key_down(KeyCode::X) {
            self.camera.move_down(speed);
        }

        if cx.key_typed(KeyCode::Return) && cx.modifiers_down(ModifierKeys::CTRL) {
            cx.toggle_fullscreen();
        }

        #[cfg(debug_assertions)]
        {
            if cx.key_typed(KeyCode::Escape) {
                cx.quit();
            }
            // TODO: Temporary
            if cx.key_typed(KeyCode::C) {
                dbg!(&self.camera);
            }
        }
    }

    /// Called every frame to render game to the screen.
    pub fn render(&mut self, delta_time: f32, cx: &mut Context) -> Result<()> {
        cx.draw_frame(RenderState {
            delta_time,
            projection: self.projection,
            view: self.camera.view(),
            // FIXME: temporary
            model: self.model,
        })?;
        Ok(())
    }

    /// Called every frame to retrieve audio samples to be played.
    pub fn audio_samples(&mut self) -> Result<Vec<f32>> {
        // TODO audio_samples https://github.com/RustAudio/cpal?
        Ok(vec![])
    }

    /// Called on every event.
    pub fn on_event(&mut self, delta_time: f32, event: Event<GameEvent>, cx: &mut Context) {
        if !cx.focused() {
            return;
        }

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(_) => self.is_dirty = true,
                WindowEvent::MouseInput {
                    button: _,
                    state: _,
                    ..
                } => {}
                WindowEvent::CloseRequested | WindowEvent::Destroyed { .. } => {
                    log::info!("shutting down...");
                    cx.quit();
                }
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                // TODO: Fixme
                DeviceEvent::MouseMotion { delta: (x, y) } => {
                    self.camera.yaw(Degrees(
                        x as f32 * self.config.mouse_sensitivity * delta_time,
                    ));
                    self.camera.pitch(Degrees(
                        y as f32 * self.config.mouse_sensitivity * delta_time,
                    ));
                }
                DeviceEvent::MouseWheel { delta } => {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * self.config.scroll_pixels_per_line,
                        MouseScrollDelta::PixelDelta(position) => position.y as f32,
                    };
                    self.camera.zoom(Degrees(y));
                    self.is_dirty = true;
                }
                DeviceEvent::Button {
                    button: _,
                    state: _,
                } => {}
                _ => (),
            },
            _ => (),
        }
    }

    /// Update the projection matrix.
    #[inline]
    fn update_projection(&mut self, cx: &Context) {
        if self.is_dirty {
            self.projection = Mat4::perspective(
                self.camera.fov().into(),
                cx.width() as f32 / cx.height() as f32,
                self.near_clip,
                self.far_clip,
            )
            .inverted_y();
            self.is_dirty = false;
        }
    }
}
