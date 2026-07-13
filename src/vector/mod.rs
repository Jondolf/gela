//! Column-major vectors.

mod gvec2;
mod gvec3;
mod gvec4;
mod swizzles;

pub use gvec2::{GVec2, gvec2};
pub use gvec3::{GVec3, gvec3};
pub use gvec4::{GVec4, gvec4};
pub use swizzles::{Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};

/// A 2-dimensional vector with `f32` components.
pub type Vec2 = GVec2<f32>;
/// A 3-dimensional vector with `f32` components.
pub type Vec3 = GVec3<f32>;
/// A 4-dimensional vector with `f32` components.
pub type Vec4 = GVec4<f32>;

/// A 2-dimensional vector with `f64` components.
pub type DVec2 = GVec2<f64>;
/// A 3-dimensional vector with `f64` components.
pub type DVec3 = GVec3<f64>;
/// A 4-dimensional vector with `f64` components.
pub type DVec4 = GVec4<f64>;

/// A 2-dimensional vector with `i32` components.
pub type IVec2 = GVec2<i32>;
/// A 3-dimensional vector with `i32` components.
pub type IVec3 = GVec3<i32>;
/// A 4-dimensional vector with `i32` components.
pub type IVec4 = GVec4<i32>;

/// A 2-dimensional vector with `u32` components.
pub type UVec2 = GVec2<u32>;
/// A 3-dimensional vector with `u32` components.
pub type UVec3 = GVec3<u32>;
/// A 4-dimensional vector with `u32` components.
pub type UVec4 = GVec4<u32>;

/// A 2-dimensional vector with `bool` components.
pub type BVec2 = GVec2<bool>;
/// A 3-dimensional vector with `bool` components.
pub type BVec3 = GVec3<bool>;
/// A 4-dimensional vector with `bool` components.
pub type BVec4 = GVec4<bool>;
