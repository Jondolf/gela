# `gela` 🍨

Generic linear algebra for games and graphics.

## Features

- Generic element types using [`gnum`]
- Vectors: `GVec2`, `GVec3`, and `GVec4` (real numbers, integers, booleans)
- Square matrices: `GMat2`, `GMat3`, `GMat4` (real numbers)
- Rotations: `GRot2`, `GRot3` (real numbers)
- Isometries: `GIso2`, `GIso3` (real numbers)
- Affine transformations: `GAffine2`, `GAffine3` (real numbers)
- Ergonomic type aliases like `Vec3` (`f32`) and `DVec3` (`f64`)
- Narrow [SIMD] ([Aos][AoSoA]) with types like `Vec3A`
- Wide [SIMD] ([SoA][AoSoA]) with types like `Vec3x4`
- Cross-platform determinism
- `no_std` support

[`gnum`]: https://crates.io/crates/gnum
[SIMD]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA

## Table of Contents

- [Getting Started](#getting-started)
- [Generic Numerics](#generic-numerics)
- [SIMD](#wide-simd)
    - [Narrow SIMD](#narrow-simd)
    - [Wide SIMD](#wide-simd)
    - [Example: Ray-Sphere Intersections](#example-ray-sphere-intersections)
    - [Generic SIMD](#generic-simd)
    - [SIMD Backends](#simd-backends)
- [Cross-Platform Determinism](#cross-platform-determinism)
- [Feature Flags](#feature-flags)

## Getting Started

Add `gela` to your `Cargo.toml`:

```toml
[dependencies]
# Replace `*` with the latest version
gela = "*"
```

Now, you can start writing math! A simple example is integrating the equations
of motion for some game objects:

```rust
use gela::vectors::Vec3;

// This method is known as semi-implicit Euler integration
// https://gafferongames.com/post/integration_basics/
fn integrate_movement(
    pos: &mut Vec3,
    vel: &mut Vec3,
    acc: Vec3,
    dt: f32,
) {
    *vel += acc * dt;
    *pos += *vel * dt;
}
```

Things get interesting when we start using the generic features provided by `gela`.
For this, you'll also want to add the [`gnum`] crate.

## Generic Numerics

`gela` uses [`gnum`] for generic numeric traits like `Num`, `Real`, `Int`, and `Float`.
This makes it possible to write highly reusable math code, like this generic axis-aligned
bounding box type:

```rust
use gela::vector::GVec3;
use gnum::Num;

// T can be an f32, f64, or even an integer type!
struct Aabb<T: Num> {
    min: GVec3<T>,
    max: Gvec3<T>,
}

impl<T: Num> Aabb {
    fn intersects(&self, other: Aabb<T>) -> T::Bool {
        self.min.num_le(other.min) & self.max.num_ge(other.max)
    }
}
```

The above is still fairly standard stuff. Where `gela` starts to stand out
is its support of SIMD, and being generic over scalar and SIMD math.

## SIMD

SIMD stands for [_Single Instruction, Multiple Data_][SIMD]. It allows you to perform
the same operation on multiple pieces of data at the same time, which can often
lead to significant performance improvements for numerical computations.

SIMD comes in two flavors: **wide SIMD** ([SoA][AoSoA]), where multiple work units (ex: vectors)
are processed simultaneously across SIMD lanes, and **narrow SIMD** ([AoS][AoSoA]),
where for example a 3D vector is put into a SIMD register, and standard vector math
is expressed using SIMD intrinsics. `gela` supports both flavors!

### Narrow SIMD

**Narrow SIMD**, also known as "horizontal" SIMD, is supported with SIMD-aligned types
like `Vec3A`, where a single vector is held in one SIMD register, and mathematical
operations exploit SIMD intrinsics wherever possible. This makes some common methods faster,
while allowing programs to be written in typical [Array of Structures (AoS)][AoSoA] fashion:

```rust
use gela::vector::Vec3A;

// Vec3A stores its components in one 128-bit SIMD register
struct PhysicsObject {
    position: Vec3A,
    velocity: Vec3A,
    mass: f32,
}
```

One downside is that storing a 128-bit SIMD register requires 16-byte alignment,
which results in wasted padding on some types:

| Type    | `f32` bytes | Align bytes | Size bytes | Padding bytes |
| ------- | ----------- | ----------- | ---------- | ------------- |
| `Vec3`  | 12          | 4           | 12         | 0             |
| `Vec3A` | 12          | 16          | 16         | 4             |
| `Mat3`  | 36          | 4           | 36         | 0             |
| `Mat3A` | 36          | 16          | 48         | 12            |

Additionally, methods that do not benefit from SIMD intrinsics can be slower,
as pulling individual elements out of SIMD registers is more expensive than direct
field access. Always measure whether the SIMD-aligned types are actually faster
for your application.

This approach to SIMD is inspired by [`glam`].

[`glam`]: https://github.com/bitshifter/glam-rs

### Wide SIMD

**Wide SIMD**, also known as "vertical" SIMD, lays out data in [SoA (Structures of Arrays)][AoSoA]
fashion, where for example an `f32x4` stores four `f32` values in one 128-bit SIMD register,
and a `Vec3x4` stores an `f32x4` for each of its coordinates:

```rust
use gela::vector::Vec3x4;
use gimd::f32x4; // See "SIMD Backends" section

struct PhysicsObjectWide {
    position: Vec3x4,
    velocity: Vec3x4,
    mass: f32x4,
}
```

This allows algorithms to operate on several physics objects at once,
providing a high level of vectorization.

Wide SIMD can provide much greater performance gains than narrow SIMD,
but often requires restructuring algorithms to operate in batches
and have minimal branching. This is demonstrated in more detail
in the following example.

### Example: Ray-Sphere Intersections

_This section is inspired by [`ultraviolet`]._

Consider an algorithm that computes the distance at which a ray intersects a sphere,
assuming that `ray_dir` is normalized. With scalar code, it looks something like this:

```rust
fn ray_sphere(
    ray_origin: Vec3,
    ray_dir: Vec3,
    sphere_origin: Vec3,
    sphere_radius_squared: f32,
) -> f32 {
    let offset = ray_origin - sphere_origin;
    let half_b = offset.dot(ray_dir);
    let c = offset.length_squared() - sphere_radius_squared;
    let discriminant = half_b * half_b - c;

    if discriminant > 0.0 {
        let discriminant_sqrt = discriminant.sqrt();

        let t1 = -half_b - discriminant_sqrt;
        if t1 > 0.0 {
            t1
        } else {
            let t2 = -half_b + discriminant_sqrt;
            if t2 > 0.0 { t2 } else { f32::MAX }
        }
    } else {
        f32::MAX
    }
}
```

In search of better performance for your software ray-tracer, you turn to SIMD to see
whether you can vectorize the algorithm. One option is to use _narrow SIMD_ via `Vec3A`:

```rust
fn ray_sphere(
    ray_origin: Vec3A,
    ray_dir: Vec3A,
    sphere_origin: Vec3A,
    sphere_radius_squared: f32,
) -> f32 {
    todo!()
}
```

This requires no other changes to the algorithm, but only helps by a small amount.
You want to go even further. With _wide SIMD_, you could cast 4 different rays
at 4 different spheres at a time!

The first step is to convert the types to their wide versions:

```rust
// Note: Vec3x4 is a type alias for GVec3<f32x4>
fn ray_sphere(
    ray_origin: Vec3x4,
    ray_dir: Vec3x4,
    sphere_origin: Vec3x4,
    sphere_radius_squared: f32x4,
) -> f32x4 {
    todo!()
}
```

Simple enough! The next four lines also remain unchanged:

```rust
let offset = ray_origin - sphere_origin;
let half_b = offset.dot(ray_dir);
let c = offset.length_squared() - sphere_radius_squared;
let discriminant = half_b * half_b - c;
```

The tricky part is the branching in this section:

```rust
// How do we express this with SIMD?
if discriminant > 0.0 {
    let discriminant_sqrt = discriminant.sqrt();

    let t1 = -half_b - discriminant_sqrt;
    // Or this?
    if t1 > 0.0 {
        t1
    } else {
        let t2 = -half_b + discriminant_sqrt;
        // Or what about this?
        if t2 > 0.0 { t2 } else { f32::MAX }
    }
} else {
    f32::MAX
}
```

We cannot use if-statements directly with SIMD, because for some lanes, the condition
could be true, while for others it may be false. The common solution is to compute the results
for both branches, and then use masks to select the correct values for each SIMD lane based on
the comparison. The vectorized version of the above code ends up looking like this:

```rust
// Outer condition
let is_discriminant_positive = discriminant.num_gt(f32x4::ZERO);
let discriminant_sqrt = discriminant.sqrt();

// Inner condition with t1, combined with outer condition
let t1 = -half_b - discriminant_sqrt;
let is_t1_valid = t1.num_gt(f32x4::ZERO) & is_discriminant_positive;

// Inner condition with t2, combined with outer condition
let t2 = -half_b + discriminant_sqrt;
let is_t2_valid = t2.num_gt(f32x4::ZERO) & is_discriminant_positive;

// Select the results matching the conditions
let t = is_t2_valid.select(t2, f32x4::MAX);
is_t1_valid.select(t1, t)
```

Finally, we have the complete algorithm:

```rust
fn ray_sphere(
    ray_origin: Vec3x4,
    ray_dir: Vec3x4,
    sphere_origin: Vec3x4,
    sphere_radius_squared: f32x4,
) -> f32x4 {
    let offset = ray_origin - sphere_origin;
    let half_b = offset.dot(ray_dir);
    let c = offset.length_squared() - sphere_radius_squared;
    let discriminant = half_b * half_b - c;

    let is_discriminant_positive = discriminant.num_gt(f32x4::ZERO);
    let discriminant_sqrt = discriminant.sqrt();

    let t1 = -half_b - discriminant_sqrt;
    let is_t1_valid = t1.num_gt(f32x4::ZERO) & is_discriminant_positive;

    let t2 = -half_b + discriminant_sqrt;
    let is_t2_valid = t2.num_gt(f32x4::ZERO) & is_discriminant_positive;

    let t = is_t2_valid.select(t2, f32x4::MAX);
    is_t1_valid.select(t1, t)
}
```

The vectorized code is actually fewer lines (19) than the original (23), but arguably
a bit harder to follow. Effective vectorization often requires restructuring your algorithms
in this way to replace branches with masks, or to eliminate them altogether.

This example used concrete types like `f32x4`. But what if you wanted to be generic over SIMD targets,
or even generalize across both scalar and SIMD math?

[`ultraviolet`]: https://github.com/fu5ha/ultraviolet

### Generic SIMD

[Remember](#generic-numerics) the numeric traits provided by [`gnum`], like `Num`, `Real`,
`Int`, and `Float`? They are also implemented for SIMD types! Any generic algorithm
that you write using them will _just work_ for both scalar and SIMD math types.

Our previous vectorized ray-sphere intersection algorithm looks like this with generic types:

```rust
fn ray_sphere<T: Real>(
    ray_origin: GVec3<T>,
    ray_dir: GVec3<T>,
    sphere_origin: GVec3<T>,
    sphere_radius_squared: T,
) -> T {
    let offset = ray_origin - sphere_origin;
    let half_b = offset.dot(ray_dir);
    let c = offset.length_squared() - sphere_radius_squared;
    let discriminant = half_b * half_b - c;

    let is_discriminant_positive = discriminant.num_gt(T::ZERO);
    let discriminant_sqrt = discriminant.sqrt();

    let t1 = -half_b - discriminant_sqrt;
    let is_t1_valid = t1.num_gt(T::ZERO) & is_discriminant_positive;

    let t2 = -half_b + discriminant_sqrt;
    let is_t2_valid = t2.num_gt(T::ZERO) & is_discriminant_positive;

    let t = is_t2_valid.select(t2, T::MAX);
    is_t1_valid.select(t1, t)
}
```

The _only_ change we had to make is to replace the vector and float types with generic versions
using a `Real` number `T`. Now, the algorithm works for `f32`, `f64`, `f32x4`, `f32x8`, `f64x2`,
or any other type that implements `Real`. Lovely!

You might be wondering whether all generic code now needs to use masks instead of simple branches.
The answer is no: for scalar code, you may use traits like `ScalarReal` instead, which makes
conditions return `bool` values like normal.

```rust
// This cannot be used with SIMD types, but it can use branches and booleans like normal.
fn ray_sphere<T: ScalarReal>(
    ray_origin: GVec3<T>,
    ray_dir: GVec3<T>,
    sphere_origin: GVec3<T>,
    sphere_radius_squared: T,
) -> T {
    let offset = ray_origin - sphere_origin;
    let half_b = offset.dot(ray_dir);
    let c = offset.length_squared() - sphere_radius_squared;
    let discriminant = half_b * half_b - c;

    if discriminant > T::ZERO {
        let discriminant_sqrt = discriminant.sqrt();

        let t1 = -half_b - discriminant_sqrt;
        if t1 > T::ZERO {
            t1
        } else {
            let t2 = -half_b + discriminant_sqrt;
            if t2 > T::ZERO { t2 } else { T::MAX }
        }
    } else {
        T::MAX
    }
}
```

Note that the [narrow SIMD](#narrow-simd) types such as `Vec3A` are concrete types
due to their specialized implementation, and do not support generic numeric types.

### SIMD Backends

Currently, three different portable SIMD implementations are officially supported by [`gnum`]
and can be used in `gela` types:

- [`wide`] provides concrete types like `f32x4` and `f64x2`, and works on both stable and nightly Rust.
- [`core::simd`] provides generic types like `Simd<T, N>` and `Mask<T, N>`, but works only on nightly Rust.
- [`gimd`] provides generic types like `Simd<T, N>` and `Mask<T, N>`, and works on both stable and nightly Rust
  by wrapping `wide` or `core::simd` types depending on features.

Type aliases like `Vec3x4` in `gela` use `gimd`, as it allows choosing between `wide`
and `core::simd` as the underlying implementation based on feature flags, works on
both stable and nightly Rust, and integrates best with `gnum`. However, the types
provided by `wide` and `core::simd` can also be used directly with the generic types
such as `GVec3<T>` if desired.

The [narrow SIMD](#narrow-simd) types also internally use `gimd`,
and choose the SIMD backend based on the `wide` and `portable_simd` features.

[`core::simd`]: https://doc.rust-lang.org/core/simd/index.html
[`wide`]: https://crates.io/crates/wide
[`gimd`]: https://crates.io/crates/gimd

## Cross-Platform Determinism

For certain applications, it can be crucial that mathematical operations return bit-identical
results across calls and platforms. Furthermore, it can also be important that scalar math
produces the exact same results as SIMD math. This way, a physics engine can produce identical
simulations across machines, while still leveraging SIMD optimizations.

`gela` supports cross-platform deterministic math with identical results across scalar and SIMD
types on all IEEE-754 compliant hardware _by default_. This is possible thanks to [`gnum`] providing
custom-made portable versions of otherwise non-deterministic methods, such as transcendental operations
(`sin`, `cos`, `atan2`, and so on). These custom methods are suffixed with `_stable`, for example
`sin_stable`. The performance difference compared to native operations is typically minimal,
or for some operations even positive, especially for SIMD types.

See the documentation of [`gnum`] for more information about its determinism guarantees.

## Feature Flags

| Feature         | Description                                                                       |
| --------------- | --------------------------------------------------------------------------------- |
| `std`           | Standard library math routines instead of portable approximations                 |
| `wide`          | [Narrow SIMD](#narrow-simd) support based on `wide`, working on stable Rust       |
| `portable_simd` | [Narrow SIMD](#narrow-simd) support based on `core::simd`, requiring nightly Rust |
| `approx`        | Approximate equality comparisons with `approx`                                    |
| `arbitrary`     | Arbitrary structured value generation with `arbitrary`                            |
| `bytecheck`     | Validation of archived types with `bytecheck`, implies `rkyv`                     |
| `bytemuck`      | Casting types to and from bytes with `bytemuck`                                   |
| `cuda`          | Alignment of types matching the requirements of CUDA                              |
| `encase`        | Writing types into and reading them from GPU buffers with `encase`                |
| `mint`          | Conversion to and from the interoperability types of `mint`                       |
| `rand`          | Random sampling of types with `rand`                                              |
| `rkyv`          | Zero-copy serialization and deserialization with `rkyv`                           |
| `serde`         | Serialization and deserialization with `serde`                                    |
| `speedy`        | Serialization and deserialization with `speedy`                                   |
| `zerocopy`      | Casting types to and from bytes with `zerocopy`                                   |

The default features are `std` and `wide`.

## Acknowledgments

The early implementation of `gela` was largely based on the delightful [`glam`] library.
You may consider `gela` to be a generic version of `glam`, with some additional features,
more officially supported determinism, and wide SIMD support.

The [`ultraviolet`] crate was also a useful reference for AoSoA-style SIMD in Rust,
and has a lovely guide on SIMD that was used as inspiration for our guide.

Finally, the [`simba`] crate was a helpful example of existing generic SIMD numerics
in the Rust ecosystem, and was an early inspiration for the [`gnum`] crate.

[`simba`]: https://github.com/dimforge/simba

## License

`gela` is free and open source. All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.
