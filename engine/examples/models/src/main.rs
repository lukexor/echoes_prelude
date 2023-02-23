use anyhow::Result;
use pix_engine::{config::Fullscreen, prelude::*, window::Positioned, Result as PixResult};
use std::io;
use tracing::Level;
use tracing_subscriber::EnvFilter;

const APPLICATION_NAME: &str = "3D Models";
const WINDOW_WIDTH: u32 = 1440;
const WINDOW_HEIGHT: u32 = 900;

const MOVEMENT_SPEED: f32 = 10.0;
const MOUSE_SENSITIVITY: f32 = 15.0;
const SCROLL_PIXELS_PER_LINE: f32 = 12.0;
const NEAR_CLIP: f32 = 0.1;
const FAR_CLIP: f32 = 100.0;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(Level::INFO.into())
                .from_env_lossy(),
        )
        .with_writer(io::stderr)
        .init();
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;

    let application = Application::initialize()?;
    let engine = Engine::builder()
        .title(APPLICATION_NAME)
        .version(env!("CARGO_PKG_VERSION"))
        .inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .positioned(Positioned::Center)
        .build();
    engine.run(application)?;
    Ok(())
}

#[derive(Debug)]
#[must_use]
struct Application {
    camera: Camera,
    projection: Mat4,
    projection_dirty: bool,
    menu_is_open: bool,
    imgui: imgui::ImGui,
}

impl Application {
    fn initialize() -> Result<Self> {
        Ok(Self {
            camera: Camera::new([0.0, 0.5, 3.0]),
            projection: Mat4::identity(),
            projection_dirty: true,
            menu_is_open: false,
            imgui: imgui::ImGui::create(),
        })
    }

    #[inline]
    fn handle_input(&mut self, cx: &mut Cx) -> Result<()> {
        if cx.key_typed(KeyCode::M) {
            self.menu_is_open = !self.menu_is_open;
        }

        if !self.menu_is_open {
            let speed = MOVEMENT_SPEED * cx.delta_time().as_secs_f32();
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

            let degrees = Degrees::new(MOUSE_SENSITIVITY * speed);
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
            cx.toggle_fullscreen(Fullscreen::Borderless);
        }

        #[cfg(debug_assertions)]
        if cx.key_typed(KeyCode::Escape) {
            cx.quit();
        }

        Ok(())
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
            NEAR_CLIP,
            FAR_CLIP,
        )
        .inverted_y();
        self.projection_dirty = false;
    }
}

pub type Cx = Context<(), RenderContext>;

impl OnUpdate for Application {
    type UserEvent = ();
    type Renderer = RenderContext;

    /// Called on engine start.
    #[inline]
    fn on_start(&mut self, cx: &mut Cx) -> PixResult<()> {
        self.imgui.initialize(cx)?;

        self.update_projection(cx.width(), cx.height());
        cx.set_projection(self.projection);
        cx.set_view(self.camera.view());

        tracing::info!("hi");

        cx.load_mesh("viking_room_mesh", "assets/meshes/viking_room.mesh");
        cx.load_texture("viking_room_texture", "assets/textures/viking_room.tx");
        cx.load_object(
            "viking_room",
            "viking_room_mesh",
            MaterialType::Texture("viking_room_texture".into()),
            Mat4::translation([-3.0, 0.0, -3.0]) * Mat4::rotation([90.0, 0.0, 70.0]),
        );

        cx.load_mesh("provence_house_mesh", "assets/meshes/provence_house.mesh");
        cx.load_texture(
            "provence_house_texture",
            "assets/textures/provence_house.tx",
        );
        cx.load_object(
            "provence_house",
            "provence_house_mesh",
            MaterialType::Texture("provence_house_texture".into()),
            Mat4::translation([5.5, 1.0, -5.0]) * Mat4::rotation([90.0, 0.0, 30.0]),
        );

        Ok(())
    }

    /// Called every frame.
    #[inline]
    fn on_update(&mut self, cx: &mut Cx) -> PixResult<()> {
        self.handle_input(cx)?;
        self.update_projection(cx.width(), cx.height());
        cx.set_projection(self.projection);
        cx.set_view(self.camera.view());
        Ok(())
    }

    /// Render UI.
    #[inline]
    fn render_imgui(&mut self, cx: &mut Cx) -> PixResult<&imgui::DrawData> {
        let ui = cx.new_ui_frame(&mut self.imgui);
        if self.menu_is_open {
            cx.set_cursor_grab(false);
            ui.show_demo_window(&mut self.menu_is_open);
        } else {
            cx.set_cursor_grab(true);
        }
        cx.end_ui_frame(ui);
        Ok(cx.render_ui_frame(&mut self.imgui))
    }

    /// Called on every event.
    #[inline]
    fn on_event(&mut self, cx: &mut Cx, event: Event<Self::UserEvent>) {
        if !cx.focused() {
            return;
        }

        self.imgui.on_event(event);
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(_) => self.projection_dirty = true,
                WindowEvent::CloseRequested | WindowEvent::Destroyed { .. } => cx.quit(),
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta: (x, y) } if !self.menu_is_open => {
                    self.camera.yaw(Degrees::new(
                        x as f32 * MOUSE_SENSITIVITY * cx.delta_time().as_secs_f32(),
                    ));
                    self.camera.pitch(Degrees::new(
                        y as f32 * MOUSE_SENSITIVITY * cx.delta_time().as_secs_f32(),
                    ));
                }
                DeviceEvent::MouseWheel { delta } if !self.menu_is_open => {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * SCROLL_PIXELS_PER_LINE,
                        MouseScrollDelta::PixelDelta(position) => position.y as f32,
                    };
                    self.camera.zoom(Degrees::new(y));
                    self.projection_dirty = true;
                }
                _ => (),
            },
            _ => (),
        }
    }
}
