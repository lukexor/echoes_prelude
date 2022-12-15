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

use echoes_engine::renderer::Shaders;
#[cfg(not(feature = "hot_reload"))]
use echoes_engine::*;
#[cfg(feature = "hot_reload")]
use hot_echoes_engine::*;

#[cfg(feature = "hot_reload")]
#[hot_lib_reloader::hot_module(
    dylib = "echoes_engine",
    // TODO: See if there's a way to make this optional
    lib_dir = concat!(env!("CARGO_TARGET_DIR"), "/debug")
)]
mod hot_echoes_engine {
    pub(crate) use echoes_engine::*;
    hot_functions_from_file!("engine/src/lib.rs");

    #[lib_change_subscription]
    pub(crate) fn _subscribe() -> hot_lib_reloader::LibReloadObserver {}
}

const APPLICATION_NAME: &str = "Echoes: Prelude";
// TODO: fullscreen?
const WINDOW_WIDTH: u32 = 1440;
const WINDOW_HEIGHT: u32 = 900;

const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/primary.vert.spv"));
const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/primary.frag.spv"));

#[tokio::main]
async fn main() -> Result<()> {
    logger::initialize()?;
    #[cfg(feature = "hot_reload")]
    initialize_logger(); // Required to properly initialize logger with hot_reload

    // TODO: tokio/reload https://github.com/rksm/hot-lib-reloader-rs/blob/master/examples/reload-events/src/main.rs

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(APPLICATION_NAME)
        .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(&event_loop)?;

    let application = Application::initialize(
        APPLICATION_NAME.into(),
        &window,
        Shaders::new(VERTEX_SHADER, FRAGMENT_SHADER),
    )?;

    main_loop(application, event_loop, window).await;

    Ok(())
}

async fn main_loop(mut engine: Application, event_loop: EventLoop<()>, window: Window) {
    log::info!("application started");

    // TODO: Need custom event loop to make it async
    event_loop.run(move |event, _window_target, control_flow| {
        control_flow.set_poll();

        log::trace!("received event: {event:?}");
        match event {
            Event::MainEventsCleared if engine.is_running() => {
                if let Err(err) = update_and_render(&mut engine, &window) {
                    log::error!("failed to render: {err}");
                    control_flow.set_exit_with_code(1);
                }
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(size) => engine.on_resized(size.width, size.height),
                WindowEvent::KeyboardInput { input, .. } => {
                    #[cfg(debug_assertions)]
                    if matches!(
                        (input.virtual_keycode, input.state),
                        (Some(VirtualKeyCode::Escape), ElementState::Pressed)
                    ) {
                        control_flow.set_exit();
                    }
                    engine.on_key_input(input);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    engine.on_mouse_input(state, button);
                }
                WindowEvent::MouseWheel { delta, phase, .. } => engine.on_mouse_wheel(delta, phase),
                WindowEvent::CursorMoved { position, .. } => {
                    engine.on_mouse_motion(position.x as f32, position.y as f32);
                }
                WindowEvent::ModifiersChanged(state) => engine.on_modifiers_changed(state),
                WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                    log::debug!("window closed or destroyed");
                    control_flow.set_exit();
                }
                _ => (),
            },
            Event::DeviceEvent { event: _, .. } => {
                // TODO: device events for controllers
            }
            Event::LoopDestroyed => {
                log::info!("shutting down...");
                control_flow.set_exit();
            }
            _ => (),
        }
    });
}
