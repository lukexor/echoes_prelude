# TODO

## Bugs

-

## Roadmap

- Engine cleanup/optimizations
- 2D images
- 2D shapes
- Text rendering
- Controller input
- Sound
- Player movement & collision
- Level/game design

## Core

- Clean up old swapchains properly
- Review <https://vkguide.dev/docs/introduction> for integration
- Add methods to load shaders, lighting
- Finish vulkan tutorial notes including major sections, options, logical
  grouping of structs, etc

- Clean up cfg compilation and try to reduce messiness.
- Add validation to Degrees/Radians - `new` returns `Option<T>`, unsafe `new_unchecked`
  returns T, has a `debug_assert`, `get` returns inner type
- Cleanup imgui integration with a trait or abstraction
- Replace winit with Windowing trait
- Make Context be constructable outside engine with a `render` method which
  returns DrawLists. Renderer is separate in engine
- Load mesh/texture early exit if already loaded
- Load mesh: name, `impl Into<DataSource>`. DataSource: File, Bytes, Network,
  Database
- Switch RenderSettings to read from config file/env. Maybe use `cvars` crate
- Put asset loader on separate thread.
- All draw functions just add to the DrawLists
- Condense time/timeEnd into a single macro
- Create separate command pools for graphics vs transfer queues and a temporary
  one for TRANSIENT
- Simplify physical device selection
- Experiment with a delete queue for vulkan objects
- Avoid println in hot loops and prefer buffered file writing
- Add FromStr impls for Vec\*, Mat4, and other level-specific data.
- Fallback textures in case resource on disk is missing
- Double check UNORM/SRGB formats for correct format
- Review supported device features and available layers across multiple platforms
- Create trait extensions for renderer based on desired features (3d, 2d, etc)
- Ensure any BufWriter calls use flush
- Ensure math operators work on normal and reference types
- Ensure pre-multiplied alpha blending
- Maybe switch to `image` crate
- Audio System
- Controller Input

## Artwork

- First screen background
- Main character sprite
- Main character animation sprites
- Ensure images are pre-multiplied alpha

## Documentation

- README with architecture
- Feature list
- Doc comments
- SAFETY comments
- hot-reloading example for engine

## Testing / Debugging

- Add assert bounds checks to hot loops to avoid index checking
- Add more assert and debug_assert checks
- Add `dhat` crate
- unit tests for matrix/quaternion operations
- Doc tests
- Integration tests
- Game editor crate
- Profiling features
- Try testing through Miri

## Release

- Check out cargo-audit
- Check out cargo-semver-check and cargo-public-api
- Build platform executables: <https://github.com/fenhl/oottracker/blob/main/crate/oottracker-utils/src/release.rs>
- Review all pub items for correct documentation, `#[must_use]`
  ,`#[non_exhausvive]`
- README
- unit tests, doctests, integration tests
- <https://github.com/EmbarkStudios/cargo-deny>
- <https://github.com/mozilla/cargo-vet>
- <https://github.com/rust-lang/rust-semverver>

## CI

- Ensure crate deps at ~0.x.x - allow warnings in CI (though leave for push hooks)
- Automate cargo update and cargo fmt
- <https://www.lurklurk.org/effective-rust/ci.html>
  - -Z minimal-versions

## Future

-
