//! Affine transformations for representing translation, rotation, scaling, and shearing.

mod gaffine2;
mod gaffine3;

pub use gaffine2::GAffine2;
pub use gaffine3::GAffine3;

/// A 2D affine transformation with `f32` components.
pub type Affine2 = GAffine2<f32>;
/// A 3D affine transformation with `f32` components.
pub type Affine3 = GAffine3<f32>;

/// A 2D affine transformation with `f64` components.
pub type DAffine2 = GAffine2<f64>;
/// A 3D affine transformation with `f64` components.
pub type DAffine3 = GAffine3<f64>;
