#![doc = include_str!("../README.md")]
#![warn(
    anonymous_parameters,
    bare_trait_objects,
    clippy::branches_sharing_code,
    clippy::map_unwrap_or,
    clippy::match_wildcard_for_single_variants,
    // clippy::missing_errors_doc,
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
    // missing_docs,
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

use std::io;

pub mod camera;
pub mod color;
pub mod config;
pub mod context;
pub mod core;
pub mod event;
#[cfg(feature = "imgui")]
pub mod imgui;
pub mod math;
pub mod matrix;
pub mod mesh;
pub mod num;
pub mod platform;
pub mod render;
pub mod scene;
pub mod shader;
pub mod vector;
pub mod window;

/// Results that can be returned from this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can be returned from this crate.
#[allow(variant_size_differences)]
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("renderer error: {0}")]
    Renderer(anyhow::Error),
    #[error("platform error: {0}")]
    Platform(anyhow::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Create a FnvHashMap.
#[macro_export]
macro_rules! hash_map {
    ($($key:expr => $value:expr),* $(,)?) => {{
        let mut map = ::fnv::FnvHashMap::with_capacity_and_hasher(4, Default::default());
        $(
        map.insert($key, $value);
        )*
        map
    }};
}

/// Create a profiling timer for a section of code.
#[macro_export]
macro_rules! time {
    ($label:ident) => {
        #[cfg(debug_assertions)]
        let mut $label = Some(::std::time::Instant::now());
    };
    (log: $label:ident) => {
        #[cfg(debug_assertions)]
        match $label {
            Some(label) => {
                ::tracing::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32())
            }
            None => tracing::warn!("Timer `{}` has been terminated.", stringify!($label)),
        };
    };
    (end: $label:ident) => {{
        #[cfg(debug_assertions)]
        match $label.take() {
            Some(label) => {
                ::tracing::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32())
            }
            None => ::tracing::warn!("Timer `{}` has been terminated.", stringify!($label)),
        };
    }};
}

pub mod prelude {
    //! Most commonly used exports for setting up an application.

    #[cfg(feature = "imgui")]
    pub use crate::imgui;
    pub use crate::{
        config::Config,
        context::Context,
        core::{Engine, OnUpdate},
        event::{
            DeviceEvent, Event, InputState, KeyCode, ModifierKeys, MouseButton, MouseScrollDelta,
            WindowEvent,
        },
        matrix::Mat4,
        mesh::MaterialType,
        num::{Degrees, Radians},
        render::Render,
        shader::{Shader, ShaderStage},
        vector::{Vec2, Vec3, Vec4},
        window::{PhysicalPosition, PhysicalSize, Position, Size},
    };

    // Macros
    pub use crate::{mat4, vec2, vec3, vec4};
}
