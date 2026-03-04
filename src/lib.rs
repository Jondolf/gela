#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(target_arch = "spirv", feature(repr_simd))]
#![cfg_attr(target_arch = "wasm64", feature(simd_wasm64))]
#![deny(
    rust_2018_compatibility,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style
)]
// clippy doesn't like `to_array(&self)`
#![allow(clippy::wrong_self_convention)]
#![cfg_attr(feature = "portable_simd", feature(portable_simd))]

#[cfg(all(
    not(feature = "std"),
    not(feature = "libm"),
    not(feature = "nostd_libm")
))]
compile_error!(
    "You must specify a math backend. Consider enabling either `std`, `libm`, or `nostd_libm`."
);

mod euler;
mod features;
mod gmat2;
mod gmat3;
mod gmat4;
mod gquat;
mod gvec2;
mod gvec3;
mod gvec4;
mod swizzles;

pub use euler::EulerRot;
pub(crate) use euler::{FromEuler, ToEuler};
pub use gmat2::{GMat2, gmat2};
pub use gmat3::{GMat3, gmat3};
pub use gmat4::{GMat4, gmat4};
pub use gquat::{GQuat, gquat};
pub use gvec2::{GVec2, gvec2};
pub use gvec3::{GVec3, gvec3};
pub use gvec4::{GVec4, gvec4};
pub use swizzles::{Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};
