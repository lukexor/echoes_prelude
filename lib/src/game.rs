//! Core game logic.

use crate::config::Config;
use anyhow::Result;
use pix_engine::{
    camera::Camera,
    config::Fullscreen,
    mesh::{Mesh, Texture, Vertex, DEFAULT_MATERIAL},
    prelude::*,
};

const VERTEX_SHADER: &str = concat!(env!("OUT_DIR"), "/primary.vert.spv");
const FRAGMENT_SHADER: &str = concat!(env!("OUT_DIR"), "/primary.frag.spv");

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
}

impl Game {
    /// Create the `Game` instance.
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: Config::new(),
            camera: Camera::new([0.0, 0.5, 3.0]),
            projection: Mat4::identity(),
            projection_dirty: true,
        })
    }

    /// Initialize the `Game` instance.
    pub async fn initialize(&mut self, cx: &mut Context<GameEvent>) -> Result<()> {
        self.update_projection(cx.width(), cx.height());

        // FIXME: tmp
        let mut vertices = vec![Vertex::default(); 3];
        vertices[0].position = vec3![1.0, 1.0, 0.0];
        vertices[1].position = vec3![-1.0, 1.0, 0.0];
        vertices[2].position = vec3![0.0, -1.0, 0.0];
        // All green
        vertices[0].color = vec3![0.0, 1.0, 0.0];
        vertices[1].color = vec3![0.0, 1.0, 0.0];
        vertices[2].color = vec3![0.0, 1.0, 0.0];

        // TODO:
        // cx.load_material();
        cx.load_mesh(Mesh::from_file("viking_room", "lib/assets/models/viking_room.obj").await?)?;
        cx.load_texture(
            Texture::from_file("viking_room", "lib/assets/textures/viking_room.png").await?,
            DEFAULT_MATERIAL,
        )?;
        cx.load_object(
            "viking_room",
            DEFAULT_MATERIAL,
            Mat4::rotation(vec3!(-90.0, 0.0, 0.0)),
        )?;

        // cx.load_mesh(
        //     Mesh::from_file("provence_house", "lib/assets/models/provence_house.obj").await?,
        // )?;
        // cx.load_texture(
        //     Texture::from_file("provence_house", "lib/assets/textures/provence_house.png").await?,
        //     DEFAULT_MATERIAL,
        // )?;
        // cx.load_object(
        //     "provence_house",
        //     DEFAULT_MATERIAL,
        //     Mat4::translation([5.0, 5.0, 0.0]) * Mat4::rotation([-90.0, 0.0, 0.0]),
        // )?;

        cx.load_mesh(Mesh::new("triangle", vertices, vec![0, 1, 2]))?;
        for x in -20..=20 {
            for y in -20..=20 {
                cx.load_object(
                    "triangle",
                    DEFAULT_MATERIAL,
                    Mat4::translation(vec3!(x as f32, 0.0, y as f32))
                        * Mat4::scale(vec3!(0.2, 0.2, 0.2)),
                )?;
            }
        }

        Ok(())
    }

    /// Called every frame to update game state.
    pub fn update(&mut self, cx: &mut Context<GameEvent>) -> Result<()> {
        // TODO: Reduce framerate when not focused.

        self.handle_input(cx);
        self.update_projection(cx.width(), cx.height());
        cx.set_projection(self.projection);
        cx.set_view(self.camera.view());

        // FIXME: temporary
        let time = cx.elapsed().as_secs_f32();
        let flash = time.sin().abs();
        cx.set_clear_color([0.0, 0.0, flash, 1.0]);
        cx.set_object_transform(
            "viking_room",
            Mat4::rotation(vec3!(-90.0, 0.0, -120.0 + time * 20.0)),
        );
        Ok(())
    }

    /// Called every frame to render game to the screen.
    pub fn render(&mut self, cx: &mut Context<GameEvent>) -> Result<()> {
        cx.draw_frame()?;
        Ok(())
    }

    fn handle_input(&mut self, cx: &mut Context<GameEvent>) {
        // FIXME: temporary
        if cx.key_typed(KeyCode::T) {
            cx.send_event(GameEvent::Debug);
        }

        let speed = 5.0 * cx.delta_time();
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
        if cx.modifiers_down(ModifierKeys::SHIFT) && cx.key_down(KeyCode::Space) {
            self.camera.move_down(speed);
        } else if cx.key_down(KeyCode::Space) {
            self.camera.move_up(speed);
        }

        if cx.key_typed(KeyCode::Return) && cx.modifiers_down(ModifierKeys::CTRL) {
            // TODO: Use config
            cx.toggle_fullscreen(Fullscreen::Borderless);
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

    /// Called every frame to retrieve audio samples to be played.
    pub fn audio_samples(&mut self) -> Result<Vec<f32>> {
        // TODO audio_samples https://github.com/RustAudio/cpal?
        Ok(vec![])
    }

    /// Called on every event.
    pub fn on_event(&mut self, cx: &mut Context<GameEvent>, event: Event<GameEvent>) {
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
                DeviceEvent::MouseMotion { delta: (x, y) } => {
                    self.camera.yaw(Degrees(
                        x as f32 * self.config.mouse_sensitivity * cx.delta_time(),
                    ));
                    self.camera.pitch(Degrees(
                        y as f32 * self.config.mouse_sensitivity * cx.delta_time(),
                    ));
                }
                DeviceEvent::MouseWheel { delta } => {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * self.config.scroll_pixels_per_line,
                        MouseScrollDelta::PixelDelta(position) => position.y as f32,
                    };
                    self.camera.zoom(Degrees(y));
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
            self.camera.fov().into(),
            width as f32 / height as f32,
            self.config.near_clip,
            self.config.far_clip,
        )
        .inverted_y();
        self.projection_dirty = false;
    }
}
