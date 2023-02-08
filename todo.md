# TODO

## Bugs

-

## Roadmap

- Review <https://vkguide.dev/docs/introduction> for integration
- Engine cleanup/optimizations
- 2D images
- 2D shapes
- Text rendering
- Controller input
- Sound
- Player movement & collision
- Level/game design

## Tasks

- 1 CORE Add validation to Degrees/Radians - `new` returns `Option<T>`, unsafe `new_unchecked` returns T, has a `debug_assert`, `get` returns inner type
- 1 CORE Refactor Mat3/Mat4 with macros to reduce duplication
- 1 CORE Experiment drop-order with shutdown method for destroying device
- 1 TEST Plug values into `glm` to generate vec/mat/quat unit tests
- 2 CORE Add imgui module for vulkan, remove extra dependency
- 2 CORE Load mesh/texture early exit if already loaded
- 2 CORE Load mesh: name, `impl Into<DataSource>`. DataSource: File, Bytes, Network, Database
- 2 FEAT Add methods to load shaders
- 2 PERF Add SIMD to vector/math operations
- 3 CORE Replace winit with Windowing trait
- 3 PERF Perform refactor/cleanup pass (incl. `cfg` conditions)
- 4 CORE Simplify physical device selection
- 4 FEAT Add lighting
- 4 PERF Clean up old swapchains properly
- 5 FEAT Switch RenderSettings to read from config file/env. Maybe use `cvars` crate
- 6 CORE Put asset loader on separate thread.
- 6 FEAT Controller Input
- 7 CORE Ensure pre-multiplied alpha blending
- 7 CORE Ensure shader snaps pixels to whole numbers
- 7 CORE Fallback textures in case resource on disk is missing
- 7 CORE Maybe switch to `image` crate
- 7 FEAT Add FromStr impls for Vec\*, Mat4, and other level-specific data.
- 8 API Explore better library error types
- 8 CORE Add more `assert` and `debug_assert` checks
- 8 FEAT Audio System
- 8 FEAT Create texture atlas/sprite sheet
- 8 PERF Add `assert` bounds checks to hot loops to avoid index checking
- 9 API Condense time/timeEnd into a single macro
- 9 API Ensure math operators work on normal and reference types
- 9 API Make example constructing EngineContext standalone with bring-your-own render loop
- 9 ART Ensure images are pre-multiplied alpha
- 9 ART First screen background
- 9 ART Main character animation sprites
- 9 ART Main character sprite
- 9 CI <https://www.lurklurk.org/effective-rust/ci.html> - -Z minimal-versions
- 9 CI Automate cargo update and cargo fmt
- 9 CI Ensure crate deps at ~0.x.x - allow warnings in CI (though leave for push hooks)
- 9 DEBUG Game editor crate
- 9 DOC Doc comments
- 9 DOC Feature list
- 9 DOC README with architecture
- 9 DOC SAFETY comments
- 9 DOC hot-reloading example for engine
- 9 EXP Experiment with a delete queue for vulkan objects
- 9 FEAT Create particle system
- 9 FEAT Profiling features
- 9 PERF Add `dhat` crate
- 9 PERF Do optimization pass. Loop unrolling, ASM checking on hot loops, SIMD
- 9 RLS <https://github.com/EmbarkStudios/cargo-deny>
- 9 RLS <https://github.com/mozilla/cargo-vet>
- 9 RLS <https://github.com/rust-lang/rust-semverver>
- 9 RLS Build platform executables: <https://github.com/fenhl/oottracker/blob/main/crate/oottracker-utils/src/release.rs>
- 9 RLS Check out cargo-audit
- 9 RLS Check out cargo-semver-check and cargo-public-api
- 9 RLS README
- 9 RLS Review all pub items for correct documentation, `#[must_use]` ,`#[non_exhausvive]`
- 9 RLS Review supported device features and available layers across multiple platforms
- 9 RLS unit tests, doctests, integration tests
- 9 TEST Doc tests
- 9 TEST Integration tests
- 9 TEST Try testing through Miri
