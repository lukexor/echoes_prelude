//! Application logic.

use anyhow::Result;
use std::{
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
pub struct State {}

impl State {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, Default, Copy, Clone)]
#[must_use]
pub struct Preferences {}

impl Preferences {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, Copy, Clone)]
#[must_use]
pub struct App {
    pub state: State,
    pub preferences: Preferences,
    pub width: u32,
    pub height: u32,

    pub start_time: Instant,
    pub last_frame_time: Instant,
    pub target_frame_rate: Duration,
    pub limit_frame_rate: bool,

    pub fps_count: usize,
    pub fps_timer: Duration,

    pub destroying: bool,
    pub minimized: bool,
    pub resized: bool,
    // pub renderer: Renderer,
}

impl App {
    pub fn create(window: &Window) -> Self {
        // TODO finish create
        let size = window.inner_size();
        Self {
            state: State::new(),
            preferences: Preferences::new(),
            width: size.width,
            height: size.height,

            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            target_frame_rate: Duration::from_secs(1) / 60,
            limit_frame_rate: false,

            fps_count: 0,
            fps_timer: Duration::default(),

            destroying: false,
            minimized: false,
            resized: false,
        }
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.destroying && !self.minimized
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            self.minimized = true;
        } else {
            self.minimized = false;
            self.resized = true;
        }
    }

    pub fn update_and_render(&mut self, _window: &Window) -> Result<()> {
        let start_time = Instant::now();
        let delta = start_time - self.last_frame_time;

        self.update(delta)?;
        self.render(delta)?;

        let end_time = Instant::now();
        let elapsed = end_time - start_time;
        self.fps_timer += delta;
        let remaining = self.target_frame_rate - elapsed;
        if !remaining.is_zero() {
            if self.limit_frame_rate {
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
        // TODO: update
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

    pub fn destroy(&mut self) {
        if self.destroying {
            return;
        }
        // TODO finish destroy
        self.destroying = true;
    }
}
