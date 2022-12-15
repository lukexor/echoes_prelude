//! Core Engine logic.

use crate::renderer::{RenderState, Renderer, Shaders};
use anyhow::Result;
use std::{
    env,
    fmt::Write,
    thread,
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
    target_fps: u32,
}

impl Config {
    pub fn new() -> Self {
        Self {
            limit_frame_rate: env::var("LIMIT_FPS").is_ok(),
            target_fps: env::var("TARGET_FPS")
                .ok()
                .and_then(|target_fps| target_fps.parse::<u32>().ok())
                .unwrap_or(60),
        }
    }
}

#[derive(Debug)]
#[must_use]
pub struct Engine {
    application_name: String,
    window_title: String,
    config: Config,

    start_time: Instant,
    last_frame_time: Instant,
    target_frame_rate: Duration,

    fps_counter: usize,
    fps_timer: Duration,
    suspended: bool,

    renderer: Renderer,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.suspended = true;
    }
}

impl Engine {
    pub fn initialize(application_name: &str, window: &Window, shaders: Shaders) -> Result<Self> {
        // TODO finish create
        let config = Config::new();
        Ok(Self {
            application_name: application_name.to_string(),
            window_title: application_name.to_string(),
            config,

            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            target_frame_rate: Duration::from_secs(1) / config.target_fps,

            fps_counter: 0,
            fps_timer: Duration::default(),

            suspended: false,

            renderer: Renderer::initialize(application_name, window, shaders)?,
        })
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.suspended
    }

    pub fn on_resized(&mut self, width: u32, height: u32) {
        log::debug!("resized event: {width}x{height}");
        if width == 0 || height == 0 {
            self.suspended = true;
        } else {
            self.suspended = false;
        }
        self.renderer.on_resized(width, height);
    }

    pub fn update_and_render(&mut self, window: &Window) -> Result<()> {
        let current_time = Instant::now();
        let delta = current_time - self.last_frame_time;

        self.update(delta)?;
        self.render(delta)?;

        let end_time = Instant::now();
        let elapsed = end_time - current_time;
        self.fps_timer += delta;
        let remaining = self
            .target_frame_rate
            .checked_sub(elapsed)
            .unwrap_or_default();
        if remaining.as_millis() > 0 {
            if self.config.limit_frame_rate {
                thread::sleep(remaining - Duration::from_millis(1));
            }
            self.fps_counter += 1;
        }

        let one_second = Duration::from_secs(1);
        if self.fps_timer > one_second {
            self.window_title.clear();
            let _ = write!(
                self.window_title,
                "{} - FPS: {}",
                self.application_name, self.fps_counter
            );
            window.set_title(&self.window_title);
            self.fps_timer -= one_second;
            self.fps_counter = 0;
        }

        self.last_frame_time = current_time;

        Ok(())
    }

    pub fn update(&mut self, _delta_time: Duration) -> Result<()> {
        // TODO: update
        Ok(())
    }

    pub fn render(&mut self, delta_time: Duration) -> Result<()> {
        // TODO: render
        self.renderer.draw_frame(RenderState { delta_time })?;
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

    pub fn on_mouse_motion(&mut self, _x: f32, _y: f32) {
        // TODO: process mouse wheel
    }

    pub fn on_axis_motion(&mut self, _axis: u32, _value: f32) {
        // TODO: process axis motion
    }

    pub fn audio_samples(&mut self) -> Result<Vec<f32>> {
        // TODO audio_samples https://github.com/RustAudio/cpal?
        Ok(vec![])
    }
}
