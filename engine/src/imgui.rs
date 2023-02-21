//! Immediate-mode GUI methods.

use crate::{prelude::*, render::RenderBackend, Error, Result};
use ::imgui::{ConfigFlags, MouseCursor};
use anyhow::anyhow;
use derive_more::{Deref, DerefMut};
use std::path::PathBuf;

pub use ::imgui::{DrawData, Ui};

#[derive(Debug, Deref, DerefMut)]
#[must_use]
pub struct ImGui {
    #[deref]
    #[deref_mut]
    pub(crate) cx: ::imgui::Context,
    pub(crate) initialized: bool,
}

impl ImGui {
    /// Initialize the imgui `Context`.
    #[inline]
    pub fn create() -> Self {
        let mut cx = ::imgui::Context::create();
        cx.set_ini_filename(None);
        // TODO: Make saving imgui data configurable
        // imgui.load_ini_settings(data);
        // if io.want_save_ini_settings {
        //     imgui.save_ini_settings(buf);
        // }
        cx.set_log_filename(PathBuf::from("logs/imgui.log"));
        cx.set_platform_name("winit".to_string());
        cx.set_renderer_name("vulkan".to_string());
        Self {
            cx,
            initialized: false,
        }
    }

    /// Initialize the imgui `Context`.
    #[inline]
    pub fn initialize<T, R: RenderBackend>(&mut self, cx: &mut Context<T, R>) -> Result<()> {
        let window = cx.window();
        let hidpi_factor = window.scale_factor().round() as f32;
        let logical_size = window.inner_size().to_logical::<f32>(hidpi_factor as f64);

        let io = self.io_mut();
        io.display_framebuffer_scale = [hidpi_factor; 2];
        io.display_size = [logical_size.width, logical_size.height];

        cx.renderer.initialize_imgui(self)?;
        self.initialized = true;

        Ok(())
    }

    #[inline]
    pub fn suspend(self) -> ::imgui::SuspendedContext {
        self.cx.suspend()
    }

    /// Called on every event.
    #[inline]
    pub fn on_event<T>(&mut self, event: Event<T>) {
        let io = self.io_mut();
        if let Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::Resized(size) => {
                    io.display_size = [size.width as f32, size.height as f32];
                }
                WindowEvent::ScaleFactorChanged {
                    scale_factor,
                    new_inner_size,
                } => {
                    io.display_framebuffer_scale = [scale_factor as f32; 2];
                    io.display_size = [new_inner_size.width as f32, new_inner_size.height as f32];
                }
                WindowEvent::KeyboardInput {
                    keycode: Some(keycode),
                    state,
                    ..
                } => {
                    if let Ok(key) = keycode.try_into() {
                        let down = state == InputState::Pressed;
                        io.add_key_event(key, down);
                    }
                }
                WindowEvent::ModifiersChanged(modifiers) => {
                    io.key_ctrl = modifiers.contains(ModifierKeys::CTRL);
                    io.key_shift = modifiers.contains(ModifierKeys::SHIFT);
                    io.key_alt = modifiers.contains(ModifierKeys::ALT);
                    io.key_super = modifiers.contains(ModifierKeys::GUI);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    io.add_mouse_pos_event([position.x as f32, position.y as f32]);
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let pixels_per_line = 12.0;
                    let (x, y) = match delta {
                        MouseScrollDelta::LineDelta(x, y) => {
                            (x * pixels_per_line, y * pixels_per_line)
                        }
                        MouseScrollDelta::PixelDelta(position) => {
                            (position.x as f32, position.y as f32)
                        }
                    };
                    io.add_mouse_wheel_event([x, y]);
                }
                WindowEvent::MouseInput { button, state, .. } => {
                    if let Ok(button) = button.try_into() {
                        let down = state == InputState::Pressed;
                        io.add_mouse_button_event(button, down);
                    }
                }
                _ => (),
            }
        }
    }
}

impl<T, R> Context<T, R> {
    /// Begin a UI frame.
    #[inline]
    pub fn new_ui_frame<'a>(&mut self, imgui: &'a mut imgui::ImGui) -> &'a mut imgui::Ui {
        use winit::dpi::PhysicalPosition;

        let io = imgui.io_mut();
        io.update_delta_time(self.delta_time());

        if io.want_set_mouse_pos {
            if let Err(err) = self
                .window()
                .set_cursor_position(PhysicalPosition::<f32>::from(io.mouse_pos))
            {
                tracing::error!("failed to set cursor position: {err:?}");
            }
        }
        imgui.new_frame()
    }

    /// Called at the end of a frame to submit updates to imgui.
    #[inline]
    pub fn end_ui_frame(&mut self, ui: &mut Ui) {
        if !ui
            .io()
            .config_flags
            .contains(ConfigFlags::NO_MOUSE_CURSOR_CHANGE)
        {
            match ui.mouse_cursor() {
                Some(cursor) if !ui.io().mouse_draw_cursor => {
                    self.window().set_cursor_visible(true);
                    self.window().set_cursor_icon(cursor_into_winit(cursor));
                }
                _ => self.window().set_cursor_visible(false),
            }
        }
    }

    /// Render a UI frame.
    #[inline]
    pub fn render_ui_frame<'a>(&mut self, imgui: &'a mut imgui::ImGui) -> &'a imgui::DrawData {
        imgui.render()
    }
}

#[inline]
pub(crate) fn cursor_into_winit(cursor: imgui::MouseCursor) -> winit::window::CursorIcon {
    use winit::window::CursorIcon;
    match cursor {
        MouseCursor::Arrow => CursorIcon::Default,
        MouseCursor::TextInput => CursorIcon::Text,
        MouseCursor::ResizeAll => CursorIcon::Move,
        MouseCursor::ResizeNS => CursorIcon::NsResize,
        MouseCursor::ResizeEW => CursorIcon::EwResize,
        MouseCursor::ResizeNESW => CursorIcon::NeswResize,
        MouseCursor::ResizeNWSE => CursorIcon::NwseResize,
        MouseCursor::Hand => CursorIcon::Hand,
        MouseCursor::NotAllowed => CursorIcon::NotAllowed,
    }
}

impl TryFrom<KeyCode> for ::imgui::Key {
    type Error = Error;

    fn try_from(keycode: KeyCode) -> std::result::Result<Self, Self::Error> {
        Ok(match keycode {
            KeyCode::Key1 => Self::Alpha1,
            KeyCode::Key2 => Self::Alpha2,
            KeyCode::Key3 => Self::Alpha3,
            KeyCode::Key4 => Self::Alpha4,
            KeyCode::Key5 => Self::Alpha5,
            KeyCode::Key6 => Self::Alpha6,
            KeyCode::Key7 => Self::Alpha7,
            KeyCode::Key8 => Self::Alpha8,
            KeyCode::Key9 => Self::Alpha9,
            KeyCode::Key0 => Self::Alpha0,
            KeyCode::A => Self::A,
            KeyCode::B => Self::B,
            KeyCode::C => Self::C,
            KeyCode::D => Self::D,
            KeyCode::E => Self::E,
            KeyCode::F => Self::F,
            KeyCode::G => Self::G,
            KeyCode::H => Self::H,
            KeyCode::I => Self::I,
            KeyCode::J => Self::J,
            KeyCode::K => Self::K,
            KeyCode::L => Self::L,
            KeyCode::M => Self::M,
            KeyCode::N => Self::N,
            KeyCode::O => Self::O,
            KeyCode::P => Self::P,
            KeyCode::Q => Self::Q,
            KeyCode::R => Self::R,
            KeyCode::S => Self::S,
            KeyCode::T => Self::T,
            KeyCode::U => Self::U,
            KeyCode::V => Self::V,
            KeyCode::W => Self::W,
            KeyCode::X => Self::X,
            KeyCode::Y => Self::Y,
            KeyCode::Z => Self::Z,
            KeyCode::Escape => Self::Escape,
            KeyCode::F1 => Self::F1,
            KeyCode::F2 => Self::F2,
            KeyCode::F3 => Self::F3,
            KeyCode::F4 => Self::F4,
            KeyCode::F5 => Self::F5,
            KeyCode::F6 => Self::F6,
            KeyCode::F7 => Self::F7,
            KeyCode::F8 => Self::F8,
            KeyCode::F9 => Self::F9,
            KeyCode::F10 => Self::F10,
            KeyCode::F11 => Self::F11,
            KeyCode::F12 => Self::F12,
            KeyCode::Snapshot => Self::PrintScreen,
            KeyCode::Insert => Self::Insert,
            KeyCode::Home => Self::Home,
            KeyCode::Delete => Self::Delete,
            KeyCode::End => Self::End,
            KeyCode::PageDown => Self::PageDown,
            KeyCode::PageUp => Self::PageUp,
            KeyCode::Left => Self::LeftArrow,
            KeyCode::Up => Self::UpArrow,
            KeyCode::Right => Self::RightArrow,
            KeyCode::Down => Self::DownArrow,
            KeyCode::Back => Self::Backspace,
            KeyCode::Return => Self::Enter,
            KeyCode::Space => Self::Space,
            KeyCode::Numpad0 => Self::Keypad0,
            KeyCode::Numpad1 => Self::Keypad1,
            KeyCode::Numpad2 => Self::Keypad2,
            KeyCode::Numpad3 => Self::Keypad3,
            KeyCode::Numpad4 => Self::Keypad4,
            KeyCode::Numpad5 => Self::Keypad5,
            KeyCode::Numpad6 => Self::Keypad6,
            KeyCode::Numpad7 => Self::Keypad7,
            KeyCode::Numpad8 => Self::Keypad8,
            KeyCode::Numpad9 => Self::Keypad9,
            KeyCode::NumpadAdd => Self::KeypadAdd,
            KeyCode::NumpadDivide => Self::KeypadDivide,
            KeyCode::NumpadDecimal => Self::KeypadDecimal,
            KeyCode::NumpadEnter => Self::KeypadEnter,
            KeyCode::NumpadEquals => Self::KeypadEqual,
            KeyCode::NumpadMultiply => Self::KeypadMultiply,
            KeyCode::NumpadSubtract => Self::KeypadSubtract,
            KeyCode::Apostrophe => Self::Apostrophe,
            KeyCode::Asterisk => Self::KeypadMultiply,
            KeyCode::Backslash => Self::Backslash,
            KeyCode::Comma => Self::Comma,
            KeyCode::Equals => Self::Equal,
            KeyCode::Grave => Self::GraveAccent,
            KeyCode::LAlt => Self::LeftAlt,
            KeyCode::LBracket => Self::LeftBracket,
            KeyCode::LControl => Self::LeftCtrl,
            KeyCode::LShift => Self::LeftShift,
            KeyCode::LWin => Self::LeftSuper,
            KeyCode::Period => Self::Period,
            KeyCode::RAlt => Self::RightAlt,
            KeyCode::RBracket => Self::RightBracket,
            KeyCode::RControl => Self::RightCtrl,
            KeyCode::RShift => Self::RightShift,
            KeyCode::RWin => Self::RightSuper,
            KeyCode::Semicolon => Self::Semicolon,
            KeyCode::Slash => Self::Slash,
            KeyCode::Tab => Self::Tab,
            _ => return Err(anyhow!("no imgui keycode mapping for {keycode:?}").into()),
        })
    }
}

impl TryFrom<MouseButton> for ::imgui::MouseButton {
    type Error = Error;

    fn try_from(button: MouseButton) -> std::result::Result<Self, Self::Error> {
        Ok(match button {
            MouseButton::Left => Self::Left,
            MouseButton::Right => Self::Right,
            MouseButton::Middle => Self::Middle,
            MouseButton::Other(index) => match index {
                0 => Self::Extra1,
                1 => Self::Extra2,
                _ => return Err(anyhow!("no imgui mouse button mapping for {button:?}").into()),
            },
        })
    }
}
