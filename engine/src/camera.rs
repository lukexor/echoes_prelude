use crate::math::{Mat4, Matrix, Radians, Vec3};

#[derive(Default, Debug, Copy, Clone)]
#[must_use]
pub struct Camera {
    position: Vec3,
    rotation: Vec3,
    view: Mat4,
    view_dirty: bool,
}

impl Camera {
    /// Create a new `Camera` at a given position.
    pub fn with_position(position: Vec3) -> Self {
        Self {
            position,
            rotation: Vec3::origin(),
            view: Matrix::translation(position).inverse(),
            view_dirty: false,
        }
    }

    /// Get the `Camera` position [Vec3].
    #[inline]
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Set the `Camera` position.
    #[inline]
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
        self.view_dirty = true;
    }

    /// Get the `Camera` rotation [Vec3].
    #[inline]
    pub fn rotation(&self) -> Vec3 {
        self.rotation
    }

    /// Set the `Camera` rotation.
    #[inline]
    pub fn set_rotation(&mut self, rotation: Vec3) {
        self.rotation = rotation;
        self.view_dirty = true;
    }

    /// Get the `Camera` view [Mat4].
    #[inline]
    pub fn view(&self) -> Mat4 {
        self.view
    }

    /// Adjust the `Camera` yaw by a given amount.
    #[inline]
    pub fn yaw(&mut self, amount: f32) {
        self.rotation.set_y(self.rotation.y() + amount);
        self.view_dirty = true;
    }

    /// Adjust the `Camera` pitch by a given amount.
    #[inline]
    pub fn pitch(&mut self, amount: f32) {
        // Clamp to avoid gimble lock
        let limit = 89f32.to_radians();
        self.rotation
            .set_x((self.rotation.x() + amount).clamp(-limit, limit));
        self.view_dirty = true;
    }

    /// Move the `Camera` forward towards the view.
    #[inline]
    pub fn move_forward(&mut self, amount: f32) {
        self.update_view();
        self.position += self.view.forward() * amount;
        self.view_dirty = true;
    }

    /// Move the `Camera` backward away from the view.
    #[inline]
    pub fn move_backward(&mut self, amount: f32) {
        self.update_view();
        self.position += self.view.backward() * amount;
        self.view_dirty = true;
    }

    /// Move the `Camera` to the left.
    #[inline]
    pub fn move_left(&mut self, amount: f32) {
        self.update_view();
        self.position += self.view.left() * amount;
        self.view_dirty = true;
    }

    /// Move the `Camera` to the right.
    #[inline]
    pub fn move_right(&mut self, amount: f32) {
        self.update_view();
        self.position += self.view.right() * amount;
        self.view_dirty = true;
    }

    /// Move the `Camera` up.
    #[inline]
    pub fn move_up(&mut self, amount: f32) {
        self.position += Vec3::up() * amount;
        self.view_dirty = true;
    }

    /// Move the `Camera` down.
    #[inline]
    pub fn move_down(&mut self, amount: f32) {
        self.position += Vec3::down() * amount;
        self.view_dirty = true;
    }

    /// Updates the view matrix, if the position or rotation has changed.
    pub fn update_view(&mut self) {
        if !self.view_dirty {
            return;
        }
        let rotation = Matrix::rotation(
            Radians(self.rotation().x()),
            Radians(self.rotation().y()),
            Radians(self.rotation().z()),
        );
        let translation = Matrix::translation(self.position());
        self.view = (rotation * translation).inverse();
        self.view_dirty = false;
    }
}
