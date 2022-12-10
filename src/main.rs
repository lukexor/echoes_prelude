#![doc = include_str!("../README.md")]
#![warn(
    anonymous_parameters,
    bare_trait_objects,
    clippy::branches_sharing_code,
    clippy::map_unwrap_or,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::needless_for_each,
    clippy::redundant_closure_for_method_calls,
    clippy::semicolon_if_nothing_returned,
    clippy::unreadable_literal,
    clippy::unwrap_used,
    clippy::expect_used,
    deprecated_in_future,
    ellipsis_inclusive_range_patterns,
    future_incompatible,
    missing_copy_implementations,
    missing_debug_implementations,
    missing_docs,
    nonstandard_style,
    rust_2018_compatibility,
    rust_2018_idioms,
    rust_2021_compatibility,
    rustdoc::bare_urls,
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags,
    rustdoc::invalid_rust_codeblocks,
    rustdoc::private_intra_doc_links,
    single_use_lifetimes,
    trivial_casts,
    trivial_numeric_casts,
    unreachable_pub,
    unused,
    variant_size_differences
)]

use anyhow::Result;
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[cfg(not(feature = "hot_reload"))]
use echoes_prelude_lib::*;
#[cfg(feature = "hot_reload")]
use hot_echoes_prelude_lib::*;

#[cfg(feature = "hot_reload")]
#[hot_lib_reloader::hot_module(
    dylib = "echoes_prelude_lib",
    // TODO: See if there's a way to make this optional
    lib_dir = concat!(env!("CARGO_TARGET_DIR"), "/debug")
)]
mod hot_echoes_prelude_lib {
    pub(crate) use echoes_prelude_lib::*;
    hot_functions_from_file!("lib/src/lib.rs");

    #[lib_change_subscription]
    pub(crate) fn _subscribe() -> hot_lib_reloader::LibReloadObserver {}
}

const APPLICATION_NAME: &str = "Echoes: Prelude";
// TODO: fullscreen?
const WINDOW_WIDTH: u32 = 1440;
const WINDOW_HEIGHT: u32 = 900;

fn create_window(event_loop: &EventLoop<()>) -> Result<Window> {
    Ok(WindowBuilder::new()
        .with_title(APPLICATION_NAME)
        .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(event_loop)?)
}

fn main_loop(mut app: App, event_loop: EventLoop<()>) {
    event_loop.run(move |event, _window_target, control_flow| {
        control_flow.set_poll();

        log::trace!("Received event: {event:?}");
        match event {
            Event::MainEventsCleared if app.is_running() => {
                if let Err(err) = update_and_render(&mut app) {
                    log::error!("Failed to render: {err}");
                    control_flow.set_exit_with_code(1);
                }
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(size) => app.on_resized(size.width, size.height),
                WindowEvent::KeyboardInput { input, .. } => {
                    #[cfg(debug_assertions)]
                    if matches!(
                        (input.virtual_keycode, input.state),
                        (Some(VirtualKeyCode::Escape), ElementState::Pressed)
                    ) {
                        control_flow.set_exit();
                    }
                    app.on_key_input(input);
                }
                WindowEvent::MouseInput { state, button, .. } => app.on_mouse_input(state, button),
                WindowEvent::MouseWheel { delta, phase, .. } => app.on_mouse_wheel(delta, phase),
                WindowEvent::CursorMoved { position, .. } => {
                    app.on_mouse_motion(position.x, position.y);
                }
                WindowEvent::ModifiersChanged(state) => app.on_modifiers_changed(state),
                WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                    log::debug!("Window closed. Shutting down...");
                    control_flow.set_exit();
                }
                _ => (),
            },
            Event::DeviceEvent { event: _, .. } => {
                // TODO: device events for controllers
            }
            Event::LoopDestroyed => {
                log::debug!("Shutting down...");
                control_flow.set_exit();
            }
            _ => (),
        }
    });
}

fn main() -> Result<()> {
    logger::initialize()?;

    // TODO: tokio/reload https://github.com/rksm/hot-lib-reloader-rs/blob/master/examples/reload-events/src/main.rs

    let event_loop = EventLoop::new();
    let window = create_window(&event_loop)?;

    // App
    let app = App::create(APPLICATION_NAME, &window)?;
    #[cfg(feature = "hot_reload")]
    initialize_logger(); // Required to properly initialize logger with reload

    main_loop(app, event_loop);

    Ok(())
}
