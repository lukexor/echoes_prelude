//! Core game logic.

use crate::config::Config;
use anyhow::Result;
use pix_engine::{
    camera::Camera,
    mesh::{Vertex, DEFAULT_MATERIAL},
    prelude::*,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GameEvent {
    Debug, // FIXME: Temporary
}

#[allow(missing_copy_implementations)]
#[derive(Debug, Clone)]
#[must_use]
pub struct Game {
    config: Config,
    camera: Camera,
    projection: Mat4,
    projection_dirty: bool,
    menu_is_open: bool,
}

impl Game {
    /// Create the `Game` instance.
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: Config::new(),
            camera: Camera::new([0.0, 0.5, 3.0]),
            projection: Mat4::identity(),
            projection_dirty: true,
            menu_is_open: false,
        })
    }

    /// Initialize the `Game` instance.
    pub fn initialize(&mut self, cx: &mut Context<'_, GameEvent>) -> Result<()> {
        self.update_projection(cx.width(), cx.height());

        cx.load_mesh("viking_room", "lib/assets/meshes/viking_room.mesh");
        cx.load_texture(
            "viking_room",
            "lib/assets/textures/viking_room.tx",
            DEFAULT_MATERIAL,
        );
        cx.load_object(
            "viking_room",
            "viking_room",
            DEFAULT_MATERIAL,
            Mat4::identity(),
        );

        cx.load_mesh("provence_house", "lib/assets/meshes/provence_house.mesh");
        cx.load_texture(
            "provence_house",
            "lib/assets/textures/provence_house.tx",
            DEFAULT_MATERIAL,
        );
        cx.load_object(
            "provence_house",
            "provence_house",
            DEFAULT_MATERIAL,
            Mat4::translation([0.0, 3.0, 0.0]) * Mat4::rotation([90.0, 0.0, 0.0]),
        );

        // FIXME: tmp
        let mut vertices = vec![Vertex::default(); 3];
        vertices[0].position = vec3![1.0, 1.0, 0.0];
        vertices[1].position = vec3![-1.0, 1.0, 0.0];
        vertices[2].position = vec3![0.0, -1.0, 0.0];
        // All green
        vertices[0].color = vec4![0.0, 1.0, 0.0, 1.0];
        vertices[1].color = vec4![0.0, 1.0, 0.0, 1.0];
        vertices[2].color = vec4![0.0, 1.0, 0.0, 1.0];
        cx.load_mesh("triangle", "lib/assets/meshes/triangle.mesh");
        for x in -20..=20 {
            for y in -20..=20 {
                cx.load_object(
                    format!("triangle_{x}_{y}"),
                    "triangle",
                    DEFAULT_MATERIAL,
                    Mat4::translation([x as f32, 0.0, y as f32]) * Mat4::scaling([0.2; 3]),
                );
            }
        }

        Ok(())
    }

    /// Called every frame to update game state.
    pub fn on_update(&mut self, cx: &mut Context<'_, GameEvent>) -> Result<()> {
        // TODO: Reduce framerate when not focused.

        self.handle_input(cx)?;
        self.update_projection(cx.width(), cx.height());
        cx.set_projection(self.projection);
        cx.set_view(self.camera.view());

        // FIXME: temporary
        let time = cx.elapsed().as_secs_f32();
        let flash = time.sin().abs();
        cx.set_clear_color([0.0, 0.0, flash, 1.0]);
        cx.set_object_transform("viking_room", Mat4::rotation([90.0, 0.0, time * 20.0]));

        if self.menu_is_open {
            cx.ui()?.show_demo_window(&mut self.menu_is_open);
            cx.set_cursor_grab(false);
        } else {
            cx.set_cursor_grab(true);
        }

        Ok(())
    }

    fn handle_input(&mut self, cx: &mut Context<'_, GameEvent>) -> Result<()> {
        // FIXME: temporary
        if cx.key_typed(KeyCode::T) {
            cx.send_event(GameEvent::Debug);
        }

        if cx.key_typed(KeyCode::M) {
            self.menu_is_open = !self.menu_is_open;
        }

        if !self.menu_is_open {
            let speed = self.config.movement_speed * cx.delta_time().as_secs_f32();
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

            let degrees = Degrees::new(self.config.mouse_sensitivity * speed);
            if cx.key_down(KeyCode::Left) {
                self.camera.yaw(-degrees);
            }
            if cx.key_down(KeyCode::Right) {
                self.camera.yaw(degrees);
            }
            if cx.key_down(KeyCode::Up) {
                self.camera.pitch(-degrees);
            }
            if cx.key_down(KeyCode::Down) {
                self.camera.pitch(degrees);
            }

            if cx.modifiers_down(ModifierKeys::SHIFT) && cx.key_down(KeyCode::Space) {
                self.camera.move_down(speed);
            } else if cx.key_down(KeyCode::Space) {
                self.camera.move_up(speed);
            }
        }

        if cx.key_typed(KeyCode::Return) && cx.modifiers_down(ModifierKeys::CTRL) {
            cx.toggle_fullscreen(self.config.fullscreen_mode);
        }

        #[cfg(debug_assertions)]
        if cx.key_typed(KeyCode::Escape) {
            cx.quit();
        }

        Ok(())
    }

    /// Called every frame to retrieve audio samples to be played.
    pub fn audio_samples(&mut self) -> Result<Vec<f32>> {
        // TODO audio_samples https://github.com/RustAudio/cpal?
        Ok(vec![])
    }

    /// Called on every event.
    pub fn on_event(&mut self, cx: &mut Context<'_, GameEvent>, event: Event<GameEvent>) {
        if !cx.focused() {
            return;
        }

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(_) => self.projection_dirty = true,
                WindowEvent::MouseInput {
                    button: _,
                    state: _,
                    ..
                } => {}
                WindowEvent::CloseRequested | WindowEvent::Destroyed { .. } => cx.quit(),
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta: (x, y) } if !self.menu_is_open => {
                    self.camera.yaw(Degrees::new(
                        x as f32 * self.config.mouse_sensitivity * cx.delta_time().as_secs_f32(),
                    ));
                    self.camera.pitch(Degrees::new(
                        y as f32 * self.config.mouse_sensitivity * cx.delta_time().as_secs_f32(),
                    ));
                }
                DeviceEvent::MouseWheel { delta } if !self.menu_is_open => {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * self.config.scroll_pixels_per_line,
                        MouseScrollDelta::PixelDelta(position) => position.y as f32,
                    };
                    self.camera.zoom(Degrees::new(y));
                    self.projection_dirty = true;
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

    /// Update the projection/view matrices.
    #[inline]
    fn update_projection(&mut self, width: u32, height: u32) {
        if !self.projection_dirty {
            return;
        }
        self.projection = Mat4::perspective(
            self.camera.fov(),
            width as f32 / height as f32,
            self.config.near_clip,
            self.config.far_clip,
        )
        .inverted_y();
        self.projection_dirty = false;
    }
}
