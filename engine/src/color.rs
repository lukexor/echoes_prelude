//! Color types and methods.

/// Convert separate RGB values to a single u32.
#[inline]
#[must_use]
pub fn rgb_to_u32(r: u32, g: u32, b: u32) -> u32 {
    ((r * 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF)
}

/// Convert a single u32 into separate RGB values.
#[inline]
#[must_use]
pub fn u32_to_rgb(value: u32) -> [u32; 3] {
    [(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF]
}
