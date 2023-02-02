use self::triangle::Tri;
use super::Render;
use crate::{
    context::Context,
    mesh::{Mesh, Vertex},
    vec2, vec3,
    vector::Vec4,
    Result,
};

mod triangle;

pub trait DrawShape {
    fn triangle(&mut self, tri: impl Into<Tri>, color: impl Into<Vec4>) -> Result<()>;
}

impl<T> DrawShape for Context<'_, T> {
    fn triangle(&mut self, tri: impl Into<Tri>, color: impl Into<Vec4>) -> Result<()> {
        let tri = tri.into();
        let color = color.into();
        let vertices = vec![
            Vertex::new(tri.p0, vec3!(), color, vec2!()),
            Vertex::new(tri.p1, vec3!(), color, vec2!()),
            Vertex::new(tri.p2, vec3!(), color, vec2!()),
        ];
        // TODO: uniquely identify mesh?
        self.load_mesh(Mesh::new("triangle", vertices, vec![0, 1, 2]));
        Ok(())
    }
}
