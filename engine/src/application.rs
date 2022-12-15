//! Core Engine logic.

use crate::{
    config::Config,
    renderer::{RenderState, Renderer, Shaders},
};
use anyhow::Result;
use std::{
    borrow::Cow,
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

#[derive(Debug)]
#[must_use]
pub struct Application {
    name: Cow<'static, str>,
    window_title: String,
    config: Config,
    last_frame_time: Instant,
    target_frame_rate: Duration,
    fps_counter: usize,
    fps_timer: Duration,
    suspended: bool,
    renderer: Renderer,
}

impl Drop for Application {
    fn drop(&mut self) {
        self.suspended = true;
    }
}

impl Application {
    pub fn initialize(name: Cow<'static, str>, window: &Window, shaders: Shaders) -> Result<Self> {
        let config = Config::new();
        let renderer = Renderer::initialize(name.as_ref(), window, shaders)?;
        let window_title = name.clone().to_string();
        Ok(Self {
            name,
            window_title,
            config,
            last_frame_time: Instant::now(),
            target_frame_rate: Duration::from_secs(1) / config.target_fps,
            fps_counter: 0,
            fps_timer: Duration::default(),
            suspended: false,
            renderer,
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
                self.name, self.fps_counter
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
