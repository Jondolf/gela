//! Column-major vectors.

mod gvec2;
mod gvec3;
mod gvec4;
mod swizzles;

pub use gvec2::{GVec2, gvec2};
pub use gvec3::{GVec3, gvec3};
pub use gvec4::{GVec4, gvec4};
pub use swizzles::{Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};
