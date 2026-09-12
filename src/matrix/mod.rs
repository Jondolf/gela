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

pub use gmat2::{GMat2, gmat2};
pub use gmat3::{GMat3, gmat3};
pub use gmat4::{GMat4, gmat4};
#[cfg(feature = "simd")]
pub use mat2a::{Mat2A, mat2a};
#[cfg(feature = "simd")]
pub use mat3a::{Mat3A, mat3a};
#[cfg(feature = "simd")]
pub use mat4a::{Mat4A, mat4a};

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
