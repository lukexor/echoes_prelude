//! Core game logic.

use anyhow::Result;
use pix_engine::{camera::Camera, math::Vec3, prelude::*, renderer::RenderState, vector};

#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct Game {
    config: Config,
    camera: Camera,
}

impl Game {
    /// Initialize the game state.
    pub fn initialize() -> Result<Self> {
        Ok(Self {
            config: Config::new(),
            camera: Camera::with_position(vector!(0.0, 0.0, 3.0)),
        })
    }

    /// Called every frame to update game state.
    pub fn update(&mut self, _delta_time: f32, cx: &mut Context) -> Result<()> {
        self.camera.update_view();
        cx.set_view(self.camera.view());
        Ok(())
    }

    /// Called every frame to render game to the screen.
    pub fn render(&mut self, delta_time: f32, cx: &mut Context) -> Result<()> {
        cx.draw_frame(RenderState {
            delta_time,
            view: self.camera.view(),
            ..Default::default()
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

        let speed = 50.0;
        let mut velocity = Vec3::origin();

        match event {
            Event::Resized(width, height) => {
                log::debug!("resized event: {width}x{height}");
            }
            Event::KeyInput { keycode, state, .. } => {
                // TODO: Keybindings
                if state == InputState::Pressed {
                    match keycode {
                        KeyCode::Escape => {
                            #[cfg(debug_assertions)]
                            cx.quit();
                        }
                        KeyCode::Q => self.camera.move_left(speed * delta_time),
                        KeyCode::E => self.camera.move_right(speed * delta_time),
                        KeyCode::W => self.camera.move_forward(speed * delta_time),
                        KeyCode::S => self.camera.move_backward(speed * delta_time),
                        KeyCode::A => self.camera.yaw(10.0 * delta_time), // Left
                        KeyCode::D => self.camera.yaw(-10.0 * delta_time), // Right
                        KeyCode::Up => self.camera.pitch(10.0 * delta_time),
                        KeyCode::Down => self.camera.pitch(-10.0 * delta_time),
                        // TODO: Temporary
                        #[cfg(debug_assertions)]
                        KeyCode::P => {
                            dbg!(&self.camera.position());
                        }
                        KeyCode::Left => (),
                        KeyCode::Right => (),
                        KeyCode::Space => self.camera.move_up(speed * delta_time),
                        KeyCode::X => self.camera.move_down(speed * delta_time),
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
            Event::MouseMotion { x: _, y: _, .. } => {}
            Event::MouseWheel { delta: _, .. } => {}
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
}
