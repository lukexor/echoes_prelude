//! Text rendering.

use crate::{context::Context, coord::Pos, Result};

impl<T, R> Context<T, R> {
    pub fn text(&mut self, pos: impl Into<Pos>, text: &str) -> Result<()> {
        todo!()
    }
}
