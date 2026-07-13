//! Rigid transformations for representing rotation followed by translation.

mod giso2;
mod giso3;

pub use giso2::GIso2;
pub use giso3::GIso3;

/// A 2D isometry with `f32` components.
pub type Iso2 = GIso2<f32>;
/// A 3D isometry with `f32` components.
pub type Iso3 = GIso3<f32>;

/// A 2D isometry with `f64` components.
pub type DIso2 = GIso2<f64>;
/// A 3D isometry with `f64` components.
pub type DIso3 = GIso3<f64>;
