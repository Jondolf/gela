//! Math library for generic linear algebra and graphics programming.
//!
//! # Feature Flags
//!
//! | Feature         | Description                                                          |
//! | --------------- | -------------------------------------------------------------------- |
//! | `approx`        | Approximate equality comparisons with [`approx`]                     |
//! | `arbitrary`     | Arbitrary structured value generation with [`arbitrary`]             |
//! | `bytecheck`     | Validation of archived types with [`bytecheck`], implies `rkyv`      |
//! | `bytemuck`      | Casting types to and from bytes with [`bytemuck`]                    |
//! | `cuda`          | Alignment of types matching the requirements of CUDA                 |
//! | `encase`        | Writing types into and reading them from GPU buffers with [`encase`] |
//! | `mint`          | Conversion to and from the interoperability types of [`mint`]        |
//! | `portable_simd` | Support for [`core::simd`] element types, requiring nightly Rust     |
//! | `rand`          | Random sampling of types with [`rand`]                               |
//! | `rkyv`          | Zero-copy serialization and deserialization with [`rkyv`]            |
//! | `serde`         | Serialization and deserialization with [`serde`]                     |
//! | `speedy`        | Serialization and deserialization with [`speedy`]                    |
//! | `zerocopy`      | Casting types to and from bytes with [`zerocopy`]                    |
//!
//! Types with a fixed minimum alignment, such as [`GVec4`], only support byte casting for
//! element types that implement `UnpaddedElement`, as smaller element types would leave
//! padding bytes in them.
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
