//! Application logic.

use crate::renderer::Renderer;
use anyhow::Result;
use std::{
    env, thread,
    time::{Duration, Instant},
};
use winit::{
    event::{
        ElementState, KeyboardInput, ModifiersState, MouseButton, MouseScrollDelta, TouchPhase,
        VirtualKeyCode,
    },
    window::Window,
};

#[derive(Debug, Default, Copy, Clone)]
#[must_use]
pub struct Config {
    limit_frame_rate: bool,
}

impl Config {
    pub fn new() -> Self {
        Self {
            limit_frame_rate: env::var("LIMIT_FPS").is_ok(),
        }
    }
}

#[derive(Debug)]
#[must_use]
pub struct App {
    pub config: Config,

    pub start_time: Instant,
    pub last_frame_time: Instant,
    pub target_frame_rate: Duration,

    pub fps_count: usize,
    pub fps_timer: Duration,

    pub suspended: bool,
    pub resized: bool,

    pub renderer: Renderer,
}

impl Drop for App {
    fn drop(&mut self) {
        self.suspended = true;
    }
}

impl App {
    pub fn create(application_name: &str, window: &Window) -> Result<Self> {
        // TODO finish create
        Ok(Self {
            config: Config::new(),

            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            target_frame_rate: Duration::from_secs(1) / 60,

            fps_count: 0,
            fps_timer: Duration::default(),

            suspended: false,
            resized: false,

            renderer: Renderer::initialize(application_name, window)?,
        })
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.suspended
    }

    pub fn on_resized(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            self.suspended = true;
        } else {
            self.suspended = false;
            self.resized = true;
        }
        self.renderer.on_resized(width, height);
    }

    pub fn update_and_render(&mut self) -> Result<()> {
        let start_time = Instant::now();
        let delta = start_time - self.last_frame_time;

        self.update(delta)?;
        self.render(delta)?;

        let end_time = Instant::now();
        let elapsed = end_time - start_time;
        self.fps_timer += delta;
        let remaining = self
            .target_frame_rate
            .checked_sub(elapsed)
            .unwrap_or_default();
        if remaining.as_millis() > 0 {
            if self.config.limit_frame_rate {
                thread::sleep(remaining - Duration::from_millis(1));
            }
            self.fps_count += 1;
        }

        let one_second = Duration::from_secs(1);
        if self.fps_timer > one_second {
            log::debug!("FPS: {}", self.fps_count);
            self.fps_timer -= one_second;
            self.fps_count = 0;
        }

        self.last_frame_time = start_time;

        Ok(())
    }

    pub fn update(&mut self, _delta: Duration) -> Result<()> {
        // TODO: update
        Ok(())
    }

    pub fn render(&mut self, _delta: Duration) -> Result<()> {
        // TODO: render
        self.renderer.draw_frame()?;
        Ok(())
    }

    pub fn on_key_input(&mut self, input: KeyboardInput) {
        // TODO: process key input
        if let Some(keycode) = input.virtual_keycode {
            match keycode {
                VirtualKeyCode::W => (),
                VirtualKeyCode::A => (),
                VirtualKeyCode::S => (),
                VirtualKeyCode::D => (),
                VirtualKeyCode::Escape => (),
                VirtualKeyCode::Up => (),
                VirtualKeyCode::Right => (),
                VirtualKeyCode::Down => (),
                VirtualKeyCode::Left => (),
                VirtualKeyCode::Space => (),
                VirtualKeyCode::LAlt => (),
                VirtualKeyCode::LControl => (),
                VirtualKeyCode::LShift => (),
                VirtualKeyCode::RAlt => (),
                VirtualKeyCode::RControl => (),
                VirtualKeyCode::RShift => (),
                _ => (),
            }
        }
    }

    pub fn on_modifiers_changed(&mut self, _state: ModifiersState) {
        // TODO: process key modifiers
    }

    pub fn on_mouse_input(&mut self, _state: ElementState, _button: MouseButton) {
        // TODO: process mouse input
    }

    pub fn on_mouse_wheel(&mut self, _delta: MouseScrollDelta, _phase: TouchPhase) {
        // TODO: process mouse wheel
    }

    pub fn on_mouse_motion(&mut self, _x: f64, _y: f64) {
        // TODO: process mouse wheel
    }

    pub fn on_axis_motion(&mut self, _axis: u32, _value: f64) {
        // TODO: process axis motion
    }

    pub fn audio_samples(&mut self) -> Result<Vec<f32>> {
        // TODO audio_samples https://github.com/RustAudio/cpal?
        Ok(vec![])
    }
}
