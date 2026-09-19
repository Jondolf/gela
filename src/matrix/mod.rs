//! Column-major matrices.

mod gmat2;
mod gmat3;
mod gmat4;
#[cfg(feature = "simd")]
mod mat2a;
#[cfg(feature = "simd")]
mod mat3a;
#[cfg(feature = "simd")]
mod mat4a;
mod symmetric_gmat2;
mod symmetric_gmat3;
mod symmetric_gmat4;

pub use gmat2::{GMat2, gmat2};
pub use gmat3::{GMat3, gmat3};
pub use gmat4::{GMat4, gmat4};
#[cfg(feature = "simd")]
pub use mat2a::{Mat2A, mat2a};
#[cfg(feature = "simd")]
pub use mat3a::{Mat3A, mat3a};
#[cfg(feature = "simd")]
pub use mat4a::{Mat4A, mat4a};
pub use symmetric_gmat2::{SymmetricGMat2, symmetric_gmat2};
pub use symmetric_gmat3::{SymmetricGMat3, symmetric_gmat3};
pub use symmetric_gmat4::{SymmetricGMat4, symmetric_gmat4};

/// A 2x2 matrix with `f32` components.
pub type Mat2 = GMat2<f32>;
/// A 3x3 matrix with `f32` components.
pub type Mat3 = GMat3<f32>;
/// A 4x4 matrix with `f32` components.
pub type Mat4 = GMat4<f32>;

/// A 2x2 matrix with `f64` components.
pub type DMat2 = GMat2<f64>;
/// A 3x3 matrix with `f64` components.
pub type DMat3 = GMat3<f64>;
/// A 4x4 matrix with `f64` components.
pub type DMat4 = GMat4<f64>;

/// A 2x2 matrix with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type Mat2x4 = GMat2<gimd::f32x4>;
/// A 3x3 matrix with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type Mat3x4 = GMat3<gimd::f32x4>;
/// A 4x4 matrix with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type Mat4x4 = GMat4<gimd::f32x4>;

/// A 2x2 matrix with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type Mat2x8 = GMat2<gimd::f32x8>;
/// A 3x3 matrix with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type Mat3x8 = GMat3<gimd::f32x8>;
/// A 4x4 matrix with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type Mat4x8 = GMat4<gimd::f32x8>;

/// A symmetric 2x2 matrix with `f32` components.
pub type SymmetricMat2 = SymmetricGMat2<f32>;
/// A symmetric 3x3 matrix with `f32` components.
pub type SymmetricMat3 = SymmetricGMat3<f32>;
/// A symmetric 4x4 matrix with `f32` components.
pub type SymmetricMat4 = SymmetricGMat4<f32>;

/// A symmetric 2x2 matrix with `f64` components.
pub type SymmetricDMat2 = SymmetricGMat2<f64>;
/// A symmetric 3x3 matrix with `f64` components.
pub type SymmetricDMat3 = SymmetricGMat3<f64>;
/// A symmetric 4x4 matrix with `f64` components.
pub type SymmetricDMat4 = SymmetricGMat4<f64>;

/// A symmetric 2x2 matrix with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type SymmetricMat2x4 = SymmetricGMat2<gimd::f32x4>;
/// A symmetric 3x3 matrix with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type SymmetricMat3x4 = SymmetricGMat3<gimd::f32x4>;
/// A symmetric 4x4 matrix with [`gimd::f32x4`] components.
#[cfg(feature = "simd")]
pub type SymmetricMat4x4 = SymmetricGMat4<gimd::f32x4>;

/// A symmetric 2x2 matrix with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type SymmetricMat2x8 = SymmetricGMat2<gimd::f32x8>;
/// A symmetric 3x3 matrix with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type SymmetricMat3x8 = SymmetricGMat3<gimd::f32x8>;
/// A symmetric 4x4 matrix with [`gimd::f32x8`] components.
#[cfg(feature = "simd")]
pub type SymmetricMat4x8 = SymmetricGMat4<gimd::f32x8>;
