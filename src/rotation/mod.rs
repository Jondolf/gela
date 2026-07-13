//! Rotation transformations.

mod euler;
mod gquat;
mod grot2;

pub use euler::EulerRot;
pub(crate) use euler::{FromEuler, ToEuler};
pub use gquat::{GQuat, gquat};
pub use grot2::{GRot2, grot2};

/// A 2D rotation with `f32` components.
pub type Rot2 = GRot2<f32>;
/// A 2D rotation with `f64` components.
pub type DRot2 = GRot2<f64>;

/// A quaternion with `f32` components.
pub type Quat = GQuat<f32>;
/// A quaternion with `f64` components.
pub type DQuat = GQuat<f64>;
