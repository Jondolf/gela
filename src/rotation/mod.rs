//! Rotation transformations.

mod euler;
mod grot2;
mod grot3;
#[cfg(feature = "simd")]
mod rot3a;

pub use euler::EulerRot;
pub(crate) use euler::{FromEuler, ToEuler};
pub use grot2::{GRot2, grot2};
pub use grot3::{GRot3, gquat};
#[cfg(feature = "simd")]
pub use rot3a::{Rot3A, rot3a};

/// A 2D rotation with `f32` components.
pub type Rot2 = GRot2<f32>;
/// A 2D rotation with `f64` components.
pub type DRot2 = GRot2<f64>;

/// A 3D rotation with `f32` components.
pub type Rot3 = GRot3<f32>;
/// A 3D rotation with `f64` components.
pub type DRot3 = GRot3<f64>;
