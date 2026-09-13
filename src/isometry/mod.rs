//! Rigid transformations for representing rotation followed by translation.

mod giso2;
mod giso3;
#[cfg(feature = "simd")]
mod iso3a;

pub use giso2::GIso2;
pub use giso3::GIso3;
#[cfg(feature = "simd")]
pub use iso3a::Iso3A;

/// A 2D isometry with `f32` components.
pub type Iso2 = GIso2<f32>;
/// A 3D isometry with `f32` components.
pub type Iso3 = GIso3<f32>;

/// A 2D isometry with `f64` components.
pub type DIso2 = GIso2<f64>;
/// A 3D isometry with `f64` components.
pub type DIso3 = GIso3<f64>;

/// A 2D isometry with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type Iso2x4 = GIso2<gimd::f32x4>;
/// A 3D isometry with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type Iso3x4 = GIso3<gimd::f32x4>;

/// A 2D isometry with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type Iso2x8 = GIso2<gimd::f32x8>;
/// A 3D isometry with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type Iso3x8 = GIso3<gimd::f32x8>;
