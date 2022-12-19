//! Core game logic.

use anyhow::Result;
use pix_engine::{prelude::*, renderer::RenderState};

#[derive(Debug)]
#[must_use]
pub struct Game {
    config: Config,
}

impl Game {
    pub fn initialize() -> Result<Self> {
        let config = Config::new();
        Ok(Self { config })
    }

    pub fn update(&mut self, _delta_time: f32, _cx: &mut Context) -> Result<()> {
        Ok(())
    }

    pub fn render(&mut self, delta_time: f32, cx: &mut Context) -> Result<()> {
        cx.draw_frame(RenderState { delta_time })?;
        Ok(())
    }

    pub fn audio_samples(&mut self) -> Result<Vec<f32>> {
        // TODO audio_samples https://github.com/RustAudio/cpal?
        Ok(vec![])
    }

    pub fn on_event(&mut self, event: Event, cx: &mut Context) {
        log::trace!("received event: {event:?}");
        match event {
            Event::Resized(width, height) => {
                log::debug!("resized event: {width}x{height}");
            }
            Event::KeyInput {
                keycode,
                state,
                modifiers: _,
                ..
            } => {
                if state == InputState::Pressed {
                    match keycode {
                        KeyCode::Escape => {
                            #[cfg(debug_assertions)]
                            cx.quit();
                        }
                        KeyCode::W => (),
                        KeyCode::A => (),
                        KeyCode::S => (),
                        KeyCode::D => (),
                        KeyCode::Up => (),
                        KeyCode::Right => (),
                        KeyCode::Down => (),
                        KeyCode::Left => (),
                        KeyCode::Space => (),
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
                modifiers: _,
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
