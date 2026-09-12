//! Column-major vectors.

mod bvec3a;
mod bvec4a;
mod gvec2;
mod gvec3;
mod gvec4;
mod swizzles;
mod vec3a;
mod vec4a;

pub use bvec3a::{BVec3A, bvec3a};
pub use bvec4a::{BVec4A, bvec4a};
pub use gvec2::{GVec2, gvec2};
pub use gvec3::{GVec3, gvec3};
pub use gvec4::{GVec4, gvec4};
pub use swizzles::{Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};
pub use vec3a::{Vec3A, vec3a};
pub use vec4a::{Vec4A, vec4a};

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

/// Four 2-dimensional vectors with `f32` components, in [AoSoA] layout.
///
/// [AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA
pub type Vec2x4 = GVec2<gimd::f32x4>;
/// Four 3-dimensional vectors with `f32` components, in [AoSoA] layout.
///
/// [AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA
pub type Vec3x4 = GVec3<gimd::f32x4>;
/// Four 4-dimensional vectors with `f32` components, in [AoSoA] layout.
///
/// [AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA
pub type Vec4x4 = GVec4<gimd::f32x4>;

/// Eight 2-dimensional vectors with `f32` components, in [AoSoA] layout.
///
/// [AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA
pub type Vec2x8 = GVec2<gimd::f32x8>;
/// Eight 3-dimensional vectors with `f32` components, in [AoSoA] layout.
///
/// [AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA
pub type Vec3x8 = GVec3<gimd::f32x8>;
/// Eight 4-dimensional vectors with `f32` components, in [AoSoA] layout.
///
/// [AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA
pub type Vec4x8 = GVec4<gimd::f32x8>;
