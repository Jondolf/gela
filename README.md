# gglam

A generic version of the [`glam`] math library for games and graphics.

[`glam`]: https://github.com/bitshifter/glam-rs

## Features

- Generic number traits using `gnum`
- Vectors: `GVec2`, `GVec3`, and `GVec4` (real numbers, integers, booleans)
- Square matrices: `GMat2`, `GMat3`, `GMat4` (real numbers)
- Quaternions: `GQuat` (real numbers)
- Affine transformations: `GAffine2`, `GAffine3` (real numbers)
- [AoSoA]-style wide [SIMD] using `wide` on stable Rust or `core::simd` on nightly Rust

[AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA
[SIMD]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data

## Why?

[`glam`] is a delightfully simple and fast math library, and a great fit for games.
It uses concrete types like `Vec3`, `DVec3`, and `BVec3` to support different types of elements,
which provides a straightforward and easy-to-use API with code that is optimized for each type of vector.

However, especially in library code, there is often a need to write math code that supports
multiple numeric types, such as `f32` and `f64`. Doing this with [`glam`] requires either
significant code duplication or defining your own ad-hoc math traits.

Additionally, [`glam`] only supports "horizontal" [SIMD] where certain types like `Vec3A`
are aligned for SIMD and can use vector operations for improved performance. This certainly
has its uses, but for maximal performance, applications like physics simulations
often use "vertical" [AoSoA]-style SIMD instead, allowing operations to be performed
on several *different* vectors/matrices/quaternions simultaneously. This is supported
by crates like [`ultraviolet`] and [`nalgebra`], but not `glam`.

`gglam` addresses both of these problems by replicating `glam`'s easy-to-use APIs,
but with generic number types. You can use `GVec3<f32>`, `GVec3<f64>`, `GVec3<i16>`,
`GVec3<bool>`, or even `GVec3<f32x4>`, unlocking the ability to write highly generic
math code that works with both scalar and SIMD types.

While `gglam` can be used fully standalone, it is not intended as a replacement for `glam`.
The concrete APIs and SIMD-aligned types of `glam` can still be valuable even alongside `gglam`.
The main purpose of `gglam` is to give users the ability to drop down to the generic APIs
or wide SIMD support when needed, even if `glam` is used elsewhere.

[`ultraviolet`]: https://github.com/fu5ha/ultraviolet
[`nalgebra`]: https://nalgebra.rs/

## License

`gglam` is free and open source. All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.
