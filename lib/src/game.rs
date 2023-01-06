//! Core game logic.

use crate::config::Config;
use anyhow::Result;
use pix_engine::{camera::Camera, prelude::*};

#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct Game {
    focused: bool,
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
            focused: false,
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
    pub fn update(&mut self, _delta_time: f32, cx: &mut Context) -> Result<()> {
        // TODO: Reduce framerate when not focused.

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
    pub fn on_event(&mut self, delta_time: f32, event: Event, cx: &mut Context) {
        log::trace!("received event: {event:?}");

        if let Event::Focused(focused) = event {
            self.focused = focused;
        }
        if !self.focused {
            return;
        }

        let speed = 40.0 * delta_time;

        match event {
            Event::Resized(width, height) => {
                log::debug!("resized event: {width}x{height}");
                self.is_dirty = true;
                self.update_projection(cx);
            }
            Event::Focused(focused) => self.focused = focused,
            Event::KeyInput { keycode, state, .. } => {
                // TODO: Keybindings
                if state == InputState::Pressed {
                    match keycode {
                        KeyCode::Escape => {
                            #[cfg(debug_assertions)]
                            cx.quit();
                        }
                        KeyCode::Q => self.camera.move_left(speed),
                        KeyCode::E => self.camera.move_right(speed),
                        KeyCode::W => self.camera.move_forward(speed),
                        KeyCode::S => self.camera.move_backward(speed),
                        KeyCode::A => self.camera.yaw(Degrees(2.0 * speed)),
                        KeyCode::D => self.camera.yaw(Degrees(-2.0 * speed)),
                        KeyCode::Up => self.camera.pitch(Degrees(2.0 * speed)),
                        KeyCode::Down => self.camera.pitch(Degrees(-2.0 * speed)),
                        // TODO: Temporary
                        #[cfg(debug_assertions)]
                        KeyCode::C => {
                            dbg!(&self.camera);
                        }
                        KeyCode::Left => (),
                        KeyCode::Right => (),
                        KeyCode::Space => self.camera.move_up(speed),
                        KeyCode::X => self.camera.move_down(speed),
                        KeyCode::LAlt => (),
                        KeyCode::LControl => (),
                        KeyCode::LShift => (),
                        KeyCode::RAlt => (),
                        KeyCode::RControl => (),
                        KeyCode::RShift => (),
                        _ => (),
                    }
                }
            }
            Event::MouseInput {
                button: _,
                state: _,
                ..
            } => {}
            Event::MouseMotion { x, y, .. } => {
                self.camera.yaw(Degrees(
                    x as f32 * self.config.mouse_sensitivity * delta_time,
                ));
                self.camera.pitch(Degrees(
                    y as f32 * self.config.mouse_sensitivity * delta_time,
                ));
            }
            Event::MouseWheel { x: _, y, .. } => {
                self.camera.zoom(Degrees(y as f32));
                self.is_dirty = true;
            }
            Event::ControllerInput {
                button: _,
                state: _,
            } => {}
            Event::Quit | Event::WindowClose { .. } => {
                log::info!("shutting down...");
                cx.quit();
            }
            _ => {
                log::trace!("{event:?} not handled");
            }
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
