# `gela` 🍨

A generic linear algebra library for games and graphics.

[`glam`]: https://github.com/bitshifter/glam-rs

## Features

- Generic element types using `gnum`
- Vectors: `GVec2`, `GVec3`, and `GVec4` (real numbers, integers, booleans)
- Square matrices: `GMat2`, `GMat3`, `GMat4` (real numbers)
- Rotations: `GRot2`, `GRot3` (real numbers)
- Isometries: `GIso2`, `GIso3` (real numbers)
- Affine transformations: `GAffine2`, `GAffine3` (real numbers)
- Ergonomic type aliases like `Vec3` (`f32`) and `DVec3` (`f64`)
- [AoSoA]-style wide [SIMD] using `wide` on stable Rust or `core::simd` on nightly Rust
- Cross-platform determinism

[AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA
[SIMD]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data

## Table of Contents

- [Getting Started](#getting-started)
- [Generic Numerics](#generic-numerics)
- [Wide SIMD](#wide-simd)
    - [Example: Ray-Sphere Intersections](#example-ray-sphere-intersections)
    - [Generic SIMD](#generic-simd)
- [Cross-Platform Determinism](#cross-platform-determinism)
- [Does it _Really_ Work?](#does-it-really-work)

## Getting Started

Add `gela` as a dependency to your `Cargo.toml`:

```toml
[dependencies]
gela = { git = "https://github.com/Jondolf/gela" }
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
For this, you'll want to also add the `gnum` crate.

## Generic Numerics

`gela` uses `gnum` for generic numeric traits like `Num`, `Real`, `Int`, and `Float`.
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
is its support of wide SIMD, and being generic over scalar and SIMD math.

## Wide SIMD

SIMD stands for [_Single Instruction, Multiple Data_][SIMD]. It allows you to perform
the same operation on multiple pieces of data at the same time, which can often
lead to significant performance improvements for numerical computations.

Some libraries like [`glam`] support "horizontal" SIMD with types like [`Vec3A`],
where calculations are still performed on one piece of data at a time, but some internal
calculations leverage SIMD instructions. This can improve performance in some cases,
while allowing code to be written in familiar [AoS (Array of Structures)][AoSoA] fashion:

```rust
use glam::Vec3A;

// Vec3A is aligned to 16 bytes, and stores a data type like __m128 (x86, SSE)
// on supported platforms, using hardware intrinsics for some operations.
struct PhysicsObject {
    position: Vec3A,
    velocity: Vec3A,
    mass: f32,
}
```

`gela` instead supports "vertical" SIMD, more commonly known as wide SIMD.
Here, data is laid out in [AoSoA (Array of Structures of Arrays)][AoSoA] fashion,
where for example an `f32x4` stores four `f32` values in one 128-bit SIMD register,
and a `Vec3x4` stores an `f32x4` for each of its coordinates:

```rust
use gela::vector::Vec3x4;
use gnum::f32x4;

struct PhysicsObjectWide {
    position: Vec3x4,
    velocity: Vec3x4,
    mass: f32x4,
}
```

Wide SIMD can provide much greater performance gains, but often requires
restructuring algorithms to be more easily vectorizable and have minimal branching.
Let's take a look at a more complicated example.

[SIMD]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[`glam`]: https://github.com/bitshifter/glam-rs
[`Vec3A`]: https://docs.rs/glam/latest/glam/f32/struct.Vec3A.html
[AoSoA]: https://en.wikipedia.org/wiki/AoS_and_SoA

### Example: Ray-Sphere Intersections

_This section is inspired by [`ultraviolet`]._

Consider an algorithm that computes the distance at which a ray intersects a sphere.
With scalar code, it looks something like this:

```rust
fn ray_sphere(
    ray_origin: Vec3,
    ray_dir: Vec3,
    sphere_origin: Vec3,
    sphere_radius_squared: f32,
) -> f32 {
    let oc = ray_origin - sphere_origin;
    let b = oc.dot(ray_dir);
    let c = oc.length_squared() - sphere_radius_squared;
    let discriminant = b * b - c;

    if discriminant > 0.0 {
        let discriminant_sqrt = discriminant.sqrt();

        let t1 = -b - discriminant_sqrt;
        if t1 > 0.0 {
            t1
        } else {
            let t2 = -b + discriminant_sqrt;
            if t2 > 0.0 { t2 } else { f32::MAX }
        }
    } else {
        f32::MAX
    }
}
```

In search of better performance for your software ray-tracer, you turn to SIMD to see
whether you can vectorize the algorithm. In particular, you want to cast 4 different rays
at 4 different spheres at a time.

The first step is to convert the types to their SIMD versions:

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
let oc = ray_origin - sphere_origin;
let b = oc.dot(ray_dir);
let c = oc.length_squared() - sphere_radius_squared;
let discriminant = b * b - c;
```

The tricky part is the branching in this section:

```rust
// How do we express this with SIMD?
if discriminant > 0.0 {
    let discriminant_sqrt = discriminant.sqrt();

    let t1 = -b - discriminant_sqrt;
    // Or this?
    if t1 > 0.0 {
        t1
    } else {
        let t2 = -b + discriminant_sqrt;
        // Or what about this?
        if t2 > 0.0 { t2 } else { f32::MAX }
    }
} else {
    f32::MAX
}
```

We cannot use if statements directly with SIMD, because for some lanes, the condition
could be true, while for others it may be false. The common solution is to essentially compute
the results for both branches, and then use masks to select the correct values for each SIMD lane
based on the comparison. The vectorized version of the above code ends up looking like this:

```rust
// Outer condition
let is_discriminant_positive = discriminant.num_gt(f32x4::ZERO);
let discriminant_sqrt = discriminant.simd_sqrt();

// Inner condition with t1, combined with outer condition
let t1 = -b - discriminant_sqrt;
let is_t1_valid = t1.num_gt(f32x4::ZERO) & is_discriminant_positive;

// Inner condition with t2, combined with outer condition
let t2 = -b + discriminant_sqrt;
let is_t2_valid = t2.num_gt(f32x4::ZERO) & is_discriminant_positive;

// Select the results matching the conditions
let t = t2.select(is_t2_valid, f32x4::splat(f32x4::MAX));
t1.select(is_t1_valid, t)
```

Finally, we have the complete algorithm:

```rust
fn ray_sphere(
    ray_origin: Vec3x4,
    ray_dir: Vec3x4,
    sphere_origin: Vec3x4,
    sphere_radius_squared: f32x4,
) -> f32x4 {
    let oc = ray_origin - sphere_origin;
    let b = oc.dot(ray_dir);
    let c = oc.length_squared() - sphere_radius_squared;
    let discriminant = b * b - c;

    let is_discriminant_positive = discriminant.num_gt(f32x4::ZERO);
    let discriminant_sqrt = discriminant.sqrt();

    let t1 = -b - discriminant_sqrt;
    let is_t1_valid = t1.num_gt(f32x4::ZERO) & is_discriminant_positive;

    let t2 = -b + discriminant_sqrt;
    let is_t2_valid = t2.num_gt(f32x4::ZERO) & is_discriminant_positive;

    let t = is_t2_valid.select(t2, f32x4::MAX);
    is_t1_valid.select(t1, t)
}
```

The vectorized code is actually fewer lines (19) than the original code (23),
but arguably a bit harder to follow. Effective vectorization often requires restructuring
your algorithms in this way to replace branches with masks, or to eliminate them altogether.

In theory, the vectorized code has to do more work, as it unconditionally evaluates _both_ branches.
Still, it ends up being faster in practice thanks to the use of SIMD instructions. On a 13th Gen Intel
Core i7-13700F processor, for 12k rays cast against 12k spheres, I get the following numbers:

TODO

This example used concrete types like `f32x4`. But what if you wanted to be generic over SIMD targets,
or even generalize across both scalar and SIMD math?

[`ultraviolet`]: https://github.com/fu5ha/ultraviolet

### Generic SIMD

[Remember](#generic-numerics) the numeric traits provided by `gnum`, like `Num`, `Real`,
`Int`, and `Float`? They are also implemented for SIMD types! Any generic algorithm
that you write using them will _just work_ for both scalar and SIMD math types.

Our previous vectorized ray-sphere intersection algorithm looks like this with generic types:

```rust
fn ray_sphere<T: Real>(
    ray_origin: Vec3<T>,
    ray_dir: Vec3<T>,
    sphere_origin: Vec3<T>,
    sphere_radius_squared: T,
) -> T {
    let oc = ray_origin - sphere_origin;
    let b = oc.dot(ray_dir);
    let c = oc.length_squared() - sphere_radius_squared;
    let discriminant = b * b - c;

    let is_discriminant_positive = discriminant.num_gt(T::ZERO);
    let discriminant_sqrt = discriminant.sqrt();

    let t1 = -b - discriminant_sqrt;
    let is_t1_valid = t1.num_gt(T::ZERO) & is_discriminant_positive;

    let t2 = -b + discriminant_sqrt;
    let is_t2_valid = t2.num_gt(T::ZERO) & is_discriminant_positive;

    let t = is_t2_valid.select(t2, T::MAX);
    is_t1_valid.select(t1, t)
}
```

The _only_ change we had to make is replace the concrete `f32x4` type with a generic
real number `T`. Now, the algorithm works for `f32`, `f64`, `f32x4`, `f64x2`,
or any other type that implements `Real`. Lovely!

You might be wondering if all generic code now needs to use masks instead of simple branches.
The answer is no: for scalar code, you may use traits like `ScalarReal` instead.

```rust
// This cannot be used with SIMD types, but it can use branches like normal.
fn ray_sphere<T: ScalarReal>(
    ray_origin: Vec3<T>,
    ray_dir: Vec3<T>,
    sphere_origin: Vec3<T>,
    sphere_radius_squared: T,
) -> T {
    let oc = ray_origin - sphere_origin;
    let b = oc.dot(ray_dir);
    let c = oc.length_squared() - sphere_radius_squared;
    let discriminant = b * b - c;

    if discriminant > T::ZERO {
        let discriminant_sqrt = discriminant.sqrt();

        let t1 = -b - discriminant_sqrt;
        if t1 > T::ZERO {
            t1
        } else {
            let t2 = -b + discriminant_sqrt;
            if t2 > T::ZERO { t2 } else { T::MAX }
        }
    } else {
        T::MAX
    }
}
```

## Cross-Platform Determinism

For certain applications, it can be crucial that mathematical operations return bit-for-bit
identical results across calls and platforms. Furthermore, it can also be important
that scalar math produces the exact same results as SIMD math. This way, a physics engine
can produce identical simulation across machines, while still leveraging SIMD optimizations.

`gela` supports cross-platform deterministic math with identical results across scalar and SIMD
types supported by `gnum` on all IEEE-754 compliant hardware. This is achieved by using custom-made
versions of otherwise non-deterministic methods, such as transcendental operations (`sin`, `cos`,
`atan2`, and so on). These custom methods are suffixed with `_stable`, for example `sin_stable`.
The performance effect is typically minimal, or for some operations even positive.

## Does it _Really_ Work?

Whether or not this style of generic math suits you depends on the application.

I originally built `gela` and `gnum` for my physics engine [Avian] in order to optimize
the contact solver with wide SIMD. I had four major goals:

1. Math code should look as close as possible to normal Rust math with concrete types,
   with trivial trait bounds.
2. SIMD code must be able to choose the optimal target, and be generic enough
   to not require writing code manually for each target or lane count.
3. SIMD must work on both stable and nightly toolchains.
4. Scalar and SIMD math must support cross-platform determinism and have methods
   that produce identical results for all relevant operations.

No existing crate I found fulfilled all four of these. Many, _many_ iterations later,
I ended up with `gela` and `gnum`. And for my needs, they fit the mold perfectly!

[Avian]: https://github.com/avianphysics/avian

## Acknowledgments

The early implementation of `gela` was largely based on the delightful [`glam`] library.
You may consider `gela` to be a generic version of `glam`, with some additional features,
more officially supported determinism, and wide SIMD support.

The [`ultraviolet`] crate was also a useful reference for AoSoA-style SIMD in Rust,
and has a lovely guide on SIMD that was used as inspiration for our guide.

Finally, the [`simba`] crate was a helpful example of existing generic SIMD numerics
in the Rust ecosystem, and was an early inspiration for the `gnum` crate.

[`simba`]: https://github.com/dimforge/simba

## License

`gela` is free and open source. All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.
