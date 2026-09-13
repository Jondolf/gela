//! Generic linear algebra for games and graphics.
//!
//! # Types
//!
//! `gela` provides the following types for linear algebra:
//!
//! - [Vectors](vector): [`GVec2`], [`GVec3`], [`GVec4`], [`Vec3A`], [`Vec4A`], [`BVec3A`], [`BVec4A`]
//! - [Matrices](matrix): [`GMat2`], [`GMat3`], [`GMat4`], [`Mat2A`], [`Mat3A`], [`Mat4A`]
//! - [Rotations](rotation): [`GRot2`], [`GRot3`], [`Rot3A`]
//! - [Isometries](isometry): [`GIso2`], [`GIso3`], [`Iso3A`]
//! - [Affine transformations](affine): [`GAffine2`], [`GAffine3`], [`Affine2A`], [`Affine3A`]
//!
//! Types prefixed with `G` are generic over the element type, and types
//! suffixed with `A` are aligned for narrow SIMD and use SIMD instructions
//! for some operations.
//!
//! Additional type aliases for different element types are also provided
//! for convenience, such as [`Vec2`], [`DRot3`], and [`Vec3x4`].
//!
//! # Getting Started
//!
//! Add `gela` to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! # Replace `*` with the latest version
//! gela = "*"
//! ```
//!
//! Now, you can start writing math! A simple example is integrating the equations
//! of motion for some game objects:
//!
//! ```rust
//! use gela::vectors::Vec3;
//!
//! // This method is known as semi-implicit Euler integration
//! // https://gafferongames.com/post/integration_basics/
//! fn integrate_movement(
//!     pos: &mut Vec3,
//!     vel: &mut Vec3,
//!     acc: Vec3,
//!     dt: f32,
//! ) {
//!     *vel += acc * dt;
//!     *pos += *vel * dt;
//! }
//! ```
//!
//! # Feature Flags
//!
//! | Feature         | Description                                                                 |
//! | --------------- | --------------------------------------------------------------------------- |
//! | `std`           | Standard library math routines instead of portable approximations           |
//! | `wide`          | Narrow SIMD support using [`wide`] (works on stable Rust)                   |
//! | `portable_simd` | Narrow SIMD support using [`core::simd`] (requires nightly Rust)            |
//! | `approx`        | Approximate equality comparisons with [`approx`]                            |
//! | `arbitrary`     | Arbitrary structured value generation with [`arbitrary`]                    |
//! | `bytecheck`     | Validation of archived types with [`bytecheck`], implies `rkyv`             |
//! | `bytemuck`      | Casting types to and from bytes with [`bytemuck`]                           |
//! | `cuda`          | Alignment of types matching the requirements of CUDA                        |
//! | `encase`        | Writing types into and reading them from GPU buffers with [`encase`]        |
//! | `mint`          | Conversion to and from the interoperability types of [`mint`]               |
//! | `rand`          | Random sampling of types with [`rand`]                                      |
//! | `rkyv`          | Zero-copy serialization and deserialization with [`rkyv`]                   |
//! | `serde`         | Serialization and deserialization with [`serde`]                            |
//! | `speedy`        | Serialization and deserialization with [`speedy`]                           |
//! | `zerocopy`      | Casting types to and from bytes with [`zerocopy`]                           |
//!
//! The default features are `std` and `wide`.
//!
//! [`approx`]: https://docs.rs/approx
//! [`arbitrary`]: https://docs.rs/arbitrary
//! [`bytecheck`]: https://docs.rs/bytecheck
//! [`bytemuck`]: https://docs.rs/bytemuck
//! [`encase`]: https://docs.rs/encase
//! [`mint`]: https://docs.rs/mint
//! [`rand`]: https://docs.rs/rand
//! [`rkyv`]: https://docs.rs/rkyv
//! [`serde`]: https://serde.rs
//! [`speedy`]: https://docs.rs/speedy
//! [`wide`]: https://docs.rs/wide
//! [`zerocopy`]: https://docs.rs/zerocopy
//! [`GVec4`]: crate::vector::GVec4

#![no_std]
#![cfg_attr(target_arch = "spirv", feature(repr_simd))]
#![cfg_attr(target_arch = "wasm64", feature(simd_wasm64))]
#![deny(
    rust_2018_compatibility,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style
)]
#![warn(missing_docs)]

#[cfg(all(
    feature = "simd",
    not(any(feature = "wide", feature = "portable_simd"))
))]
compile_error!(
    "The `simd` feature is internal. Enable the `wide` or `portable_simd` feature instead"
);

pub mod affine;
pub mod isometry;
pub mod matrix;
pub mod rotation;
pub mod vector;

mod deref;
mod features;

#[cfg(any(feature = "bytemuck", feature = "zerocopy"))]
pub use features::UnpaddedElement;
#[cfg(feature = "rand")]
pub use features::{UniformGVec2, UniformGVec3, UniformGVec4};
#[cfg(all(feature = "rand", feature = "simd"))]
pub use features::{UniformVec3A, UniformVec4A};

#[allow(unused_imports, reason = "used in documentation")]
use crate::prelude::*;

/// Re-exports for convenience.
pub mod prelude {
    pub use crate::{affine::*, isometry::*, matrix::*, rotation::*, vector::*};
}
