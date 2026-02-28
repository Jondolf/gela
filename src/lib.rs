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

mod features;
mod gvec2;

pub use gvec2::{GVec2, gvec2};
