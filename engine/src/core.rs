//! Core engine traits and functionality.

#[cfg(feature = "imgui")]
use crate::imgui::{ImGui, Ui};
use crate::{
    config::{Config, Fullscreen},
    prelude::*,
    render::{RenderBackend, RenderSettings, Renderer},
    window::{
        winit::{FullscreenModeExt, PositionedExt},
        Positioned, WindowCreateInfo,
    },
    Error, Result,
};
use anyhow::Context as _;
use std::{
    env,
    fmt::Debug,
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::{self, File},
    io::{AsyncReadExt, BufReader},
    runtime,
};
use winit::{
    event::{Event as WinitEvent, StartCause},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::WindowBuilder,
};

pub trait OnUpdate {
    type UserEvent: Copy + Debug + 'static;
    type Renderer: RenderBackend;

    /// Called on engine start for initializing resources and state.
    fn on_start(&mut self, _cx: &mut Context<Self::UserEvent, Self::Renderer>) -> Result<()> {
        Ok(())
    }

    /// Called every frame to update state and render frames.
    fn on_update(
        &mut self,
        cx: &mut Context<Self::UserEvent, Self::Renderer>,
        #[cfg(feature = "imgui")] ui: &mut Ui,
    ) -> Result<()>;

    /// Called on engine shutdown to clean up resources.
    fn on_stop(&mut self, _cx: &mut Context<Self::UserEvent, Self::Renderer>) {}

    /// Called on every event.
    fn on_event(
        &mut self,
        _cx: &mut Context<Self::UserEvent, Self::Renderer>,
        _event: Event<Self::UserEvent>,
    ) {
    }
}

#[derive(Default, Debug)]
#[must_use]
pub struct EngineBuilder(Engine);

impl EngineBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn title(&mut self, title: impl Into<String>) -> &mut Self {
        self.0.config.window_title = title.into();
        self
    }

    pub fn version(&mut self, version: impl Into<String>) -> &mut Self {
        self.0.version = version.into();
        self
    }

    pub fn inner_size(&mut self, size: impl Into<Size>) -> &mut Self {
        self.0.window_create_info.size = size.into();
        self
    }

    pub fn positioned(&mut self, position: impl Into<Positioned>) -> &mut Self {
        self.0.window_create_info.positioned = position.into();
        self
    }

    pub fn cursor_grab(&mut self, cursor_grab: bool) -> &mut Self {
        self.0.config.cursor_grab = cursor_grab;
        self
    }

    pub fn fullscreen(&mut self, fullscreen: impl Into<Option<Fullscreen>>) -> &mut Self {
        self.0.config.fullscreen = fullscreen.into();
        self
    }

    pub fn assets_directory(&mut self, directory: impl Into<PathBuf>) -> &mut Self {
        self.0.config.asset_directory = Some(directory.into());
        self
    }

    pub fn resizable(&mut self, resizable: bool) -> &mut Self {
        self.0.window_create_info.resizable = resizable;
        self
    }

    pub fn build(&mut self) -> Engine {
        self.0.clone()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct Engine {
    version: String,
    window_create_info: WindowCreateInfo,
    config: Config,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            version: "1.0.0".into(),
            window_create_info: WindowCreateInfo::default(),
            config: Config::default(),
        }
    }
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    pub fn run<A: OnUpdate + 'static>(self, mut app: A) -> Result<()> {
        if let Some(assets_directory) = &self.config.asset_directory {
            let runtime = runtime::Builder::new_multi_thread().enable_all().build()?;
            runtime.block_on(Self::check_convert_assets(assets_directory))?;
        }

        let event_loop = EventLoopBuilder::<A::UserEvent>::with_user_event().build();
        let event_proxy = event_loop.create_proxy();
        let mut cx = None;
        #[cfg(feature = "imgui")]
        let mut gui = ImGui::create();

        tracing::debug!("starting `Engine::on_update` loop.");
        event_loop.run(move |event, event_loop, control_flow| {
            if matches!(event, WinitEvent::NewEvents(StartCause::Init)) {
                let Ok(window) = WindowBuilder::new()
                    .with_title(&self.config.window_title)
                    .with_inner_size(self.window_create_info.size)
                    .with_position(self.window_create_info.positioned.for_monitor(event_loop.primary_monitor(), self.window_create_info.size))
                    .with_resizable(self.window_create_info.resizable)
                    // TODO: Support Exclusive
                    .with_fullscreen(self
                        .config.fullscreen.and_then(|fullscreen| fullscreen.for_monitor(event_loop.primary_monitor()))
                    )
                    .build(event_loop)
                    .context("failed to create window")
                    .map_err(|err| tracing::error!("{err:?}")) else {
                        control_flow.set_exit_with_code(1);
                        return;
                    };

                // TODO: Read from saved configuration
                let settings = RenderSettings::default();
                let Ok(renderer) = Renderer::initialize(&self.config.window_title, &self.version, &window, settings)
                    .map_err(|err| tracing::error!("{err:?}")) else {
                    control_flow.set_exit_with_code(2);
                    return;
                };
                let mut context = Context::<A::UserEvent, A::Renderer>::with_user_events(self.config.clone(), window, event_proxy.clone(), renderer);

                let on_start = app.on_start(&mut context);
                if on_start.is_err() || context.should_quit() {
                    tracing::debug!("quitting after on_start with `Engine::on_stop`");
                    app.on_stop(&mut context);
                    return match on_start {
                        Ok(_) => control_flow.set_exit(),
                        Err(err) => Self::handle_error(control_flow, 1, err),
                    }
                } else {
                    #[cfg(feature = "imgui")]
                    if let Err(err) = gui.initialize(&mut context) {
                        Self::handle_error(control_flow, 2, err);
                    }
                    cx = Some(context);
                }
            }

            if let Some(cx) = &mut cx {
                match event {
                    WinitEvent::MainEventsCleared if cx.is_running() => {
                        cx.begin_frame();
                        #[cfg(feature = "imgui")]
                        let ui = cx.begin_ui_frame(&mut gui);
                        if let Err(err) = app.on_update(cx, #[cfg(feature = "imgui")] ui) {
                            return Self::handle_error(control_flow, 10, err);
                        }

                        #[cfg(feature = "imgui")]
                        cx.end_ui_frame(ui);
                        #[cfg(feature = "imgui")]
                        let ui_data = cx.render_ui_frame(&mut gui);
                        if let Err(err) = cx.draw_frame(#[cfg(feature = "imgui")] ui_data) {
                            return Self::handle_error(control_flow, 11, err);
                        }
                        cx.end_frame();
                    }
                    _ => {
                        tracing::trace!("received event: {event:?}");
                        if let Ok(event) = event.try_into() {
                            cx.on_event(event);
                            #[cfg(feature = "imgui")]
                            gui.on_event(event);
                            app.on_event(cx, event);
                        }
                    }
                }

                if cx.should_quit() && !cx.is_quitting {
                    cx.is_quitting = true;
                    tracing::debug!("shutting down with `Engine::on_stop`");
                    app.on_stop(cx);
                    // on_stop allows aborting the quit request
                    if cx.should_quit() {
                        tracing::debug!("shutting down...");
                        control_flow.set_exit();
                    } else {
                        cx.is_quitting = false;
                    }
                }
            }
        });
    }

    async fn check_convert_assets(assets_dir: &PathBuf) -> Result<()> {
        let out_dir = PathBuf::from(env!("OUT_DIR"));

        let asset_mtime_file = out_dir.join(assets_dir).with_extension("mtime");
        let should_convert = if asset_mtime_file.exists() {
            let mut asset_mtime_reader =
                BufReader::new(File::open(&asset_mtime_file).await.with_context(|| {
                    format!("failed to open asset mtime file: {asset_mtime_file:?}")
                })?);
            let mut last_modified_timestamp = String::with_capacity(16);
            asset_mtime_reader
                .read_to_string(&mut last_modified_timestamp)
                .await?;
            let last_modified =
                Duration::from_secs(last_modified_timestamp.parse::<u64>().unwrap_or_default());

            let asset_dir_metadata = fs::metadata(assets_dir).await?;
            let current_modified = asset_dir_metadata
                .modified()?
                .duration_since(UNIX_EPOCH)
                .context("invalid unix epoch")?;

            tracing::debug!(
                "asset modified {}. last modified: {}",
                current_modified.as_secs(),
                last_modified.as_secs()
            );
            current_modified > last_modified
        } else {
            tracing::debug!("asset modified timestamp doesn't exist");
            true
        };

        if should_convert {
            asset_loader::convert_all(assets_dir, out_dir)
                .await
                .context("failed to convert assets directory")?;

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("invalid unix epoch")?
                .as_secs();
            fs::write(&asset_mtime_file, now.to_string())
                .await
                .with_context(|| {
                    format!("failed to write asset mtime file: {asset_mtime_file:?}")
                })?;
        }
        Ok(())
    }

    fn handle_error(control_flow: &mut ControlFlow, code: i32, err: Error) {
        tracing::error!("{err:?}");
        control_flow.set_exit_with_code(code);
    }
}
