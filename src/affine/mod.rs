//! Affine transformations for representing translation, rotation, scaling, and shearing.

#[cfg(feature = "simd")]
mod affine2a;
#[cfg(feature = "simd")]
mod affine3a;
mod gaffine2;
mod gaffine3;

#[cfg(feature = "simd")]
pub use affine2a::Affine2A;
#[cfg(feature = "simd")]
pub use affine3a::Affine3A;
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

/// A 2D affine transformation with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type Affine2x4 = GAffine2<gimd::f32x4>;
/// A 3D affine transformation with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type Affine3x4 = GAffine3<gimd::f32x4>;

/// A 2D affine transformation with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type Affine2x8 = GAffine2<gimd::f32x8>;
/// A 3D affine transformation with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type Affine3x8 = GAffine3<gimd::f32x8>;
