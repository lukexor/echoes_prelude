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

use thiserror::Error;

pub mod camera;
#[macro_use]
pub mod profiling;
pub mod config;
pub mod context;
pub mod core;
pub mod event;
pub mod math;
pub mod platform;
pub mod renderer;
pub mod window;

pub type Result<T> = std::result::Result<T, Error>;

#[allow(variant_size_differences)]
#[derive(Error, Debug)]
pub enum Error {
    #[error("invalid image format: bit_depth: {bit_depth:?}, color_type: {color_type:?}")]
    UnsupportedImageFormat {
        bit_depth: png::BitDepth,
        color_type: png::ColorType,
    },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub mod prelude {
    pub use crate::{
        config::Config,
        context::Context,
        core::{Engine, Update},
        event::{
            DeviceEvent, Event, InputState, KeyCode, ModifierKeys, MouseButton, MouseScrollDelta,
            WindowEvent,
        },
        math::{Degrees, Mat4, Radians, Vec2, Vec3},
        matrix,
        renderer::{RenderState, Renderer, Shader, ShaderType},
        vector,
        window::{PhysicalPosition, PhysicalSize, Position, Size},
    };
}
