//! Camera functionality.

use crate::{matrix::Mat4, num::Degrees, vec3, vector::Vec3};

const DEFAULT_YAW: Degrees = Degrees::new_unchecked(-90.0);
const DEFAULT_PITCH: Degrees = Degrees::new_unchecked(0.0);
const DEFAULT_FOV: Degrees = Degrees::new_unchecked(45.0);

const PITCH_LIMIT: Degrees = Degrees::new_unchecked(89.0);
const FOV_MIN: Degrees = Degrees::new_unchecked(1.0);
const FOV_MAX: Degrees = Degrees::new_unchecked(45.0);

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[must_use]
pub struct CameraData {
    pub projection: Mat4,
    pub view: Mat4,
    pub projection_view: Mat4,
}

#[derive(Default, Debug, Copy, Clone)]
#[must_use]
pub struct Camera {
    position: Vec3,
    target: Vec3,
    up: Vec3,
    world_up: Vec3,
    right: Vec3,
    yaw: Degrees,
    pitch: Degrees,
    fov: Degrees,
    view: Mat4,
    is_dirty: bool,
}

impl Camera {
    /// Create a new `Camera` at a given position and rotation.
    pub fn new(position: impl Into<Vec3>) -> Self {
        let position = position.into();
        let mut camera = Self {
            position,
            target: Vec3::forward(),
            up: Vec3::up(),
            world_up: Vec3::up(),
            right: Vec3::origin(),
            yaw: DEFAULT_YAW,
            pitch: DEFAULT_PITCH,
            fov: DEFAULT_FOV,
            view: Mat4::translation(position).inverse(),
            is_dirty: true,
        };
        camera.update_view();
        camera
    }

    /// Reset the `Camera`.
    pub fn reset(&mut self) {
        self.position = Vec3::origin();
        self.world_up = Vec3::up();
        self.yaw = DEFAULT_YAW;
        self.pitch = DEFAULT_PITCH;
        self.fov = DEFAULT_FOV;
        self.is_dirty = true;
        self.update_view();
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
        self.is_dirty = true;
    }

    /// Get the `Camera` Field of View (FOV).
    #[inline]
    pub fn fov(&self) -> Degrees {
        self.fov
    }

    /// Get the `Camera` view [Mat4].
    #[inline]
    pub fn view(&mut self) -> Mat4 {
        self.update_view();
        self.view
    }

    /// Angle the `Camera` left or right by a given amount.
    #[inline]
    pub fn yaw(&mut self, amount: Degrees) {
        self.yaw += amount;
        self.is_dirty = true;
    }

    /// Angle the `Camera` up or down by a given amount.
    #[inline]
    pub fn pitch(&mut self, amount: Degrees) {
        // SAFETY: We clamp the values to the correct range.
        self.pitch = Degrees::new((self.pitch - amount).clamp(-*PITCH_LIMIT, *PITCH_LIMIT));
        self.is_dirty = true;
    }

    /// Angle the `Camera` up or down by a given amount.
    #[inline]
    pub fn zoom(&mut self, amount: Degrees) {
        // SAFETY: We clamp the values to the correct range.
        self.fov = Degrees::new((self.fov - amount).clamp(*FOV_MIN, *FOV_MAX));
    }

    /// Move the `Camera` forward towards the target.
    #[inline]
    pub fn move_forward(&mut self, amount: f32) {
        self.position += amount * self.target;
        self.is_dirty = true;
    }

    /// Move the `Camera` backward away from the target.
    #[inline]
    pub fn move_backward(&mut self, amount: f32) {
        self.position -= amount * self.target;
        self.is_dirty = true;
    }

    /// Move the `Camera` to the left.
    #[inline]
    pub fn move_left(&mut self, amount: f32) {
        self.position -= amount * self.right;
        self.is_dirty = true;
    }

    /// Move the `Camera` to the right.
    #[inline]
    pub fn move_right(&mut self, amount: f32) {
        self.position += amount * self.right;
        self.is_dirty = true;
    }

    /// Move the `Camera` up.
    #[inline]
    pub fn move_up(&mut self, amount: f32) {
        self.position += self.view.up() * amount;
        self.is_dirty = true;
    }

    /// Move the `Camera` down.
    #[inline]
    pub fn move_down(&mut self, amount: f32) {
        self.position += self.view.down() * amount;
        self.is_dirty = true;
    }

    /// Updates the view matrix, if the position or rotation has changed.
    fn update_view(&mut self) {
        if !self.is_dirty {
            return;
        }
        let (yaw_sin, yaw_cos) = self.yaw.to_radians().sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.to_radians().sin_cos();
        self.target = vec3!(yaw_cos * pitch_cos, pitch_sin, yaw_sin * pitch_cos).normalized();
        self.right = self.target.cross(self.world_up).normalized();
        self.up = self.right.cross(self.target).normalized();
        self.view = Mat4::look_at(self.position, self.position + self.target, self.up);
        self.is_dirty = false;
    }
}
