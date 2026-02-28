use core::{
    iter::{Product, Sum},
    ops::*,
};

use gnum::{
    simd::{NumEq, NumOrd, Select, SimdBool},
    traits::{
        CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, DivEuclid, Float, Int, Num, Real,
        RemEuclid, SaturatingAdd, SaturatingDiv, SaturatingMul, SaturatingSub, Signed, WrappingAdd,
        WrappingDiv, WrappingMul, WrappingSub,
    },
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// Creates a 2-dimensional vector.
#[inline(always)]
#[must_use]
pub const fn gvec2<T: Copy>(x: T, y: T) -> GVec2<T> {
    GVec2::new(x, y)
}

/// A 2-dimensional vector.
#[derive(Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[cfg_attr(
    feature = "zerocopy",
    derive(FromBytes, Immutable, IntoBytes, KnownLayout)
)]
#[cfg_attr(feature = "cuda", repr(align(8)))]
#[repr(C)]
#[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
pub struct GVec2<T: Copy> {
    /// The X component of the vector.
    pub x: T,
    /// The Y component of the vector.
    pub y: T,
}

/// # Basic Number Constants
impl<T: Num> GVec2<T> {
    /// All zeros.
    pub const ZERO: Self = Self::new(T::ZERO, T::ZERO);

    /// All ones.
    pub const ONE: Self = Self::new(T::ONE, T::ONE);

    /// All `MIN`.
    pub const MIN: Self = Self::new(T::MIN, T::MIN);

    /// All `MAX`.
    pub const MAX: Self = Self::new(T::MAX, T::MAX);

    /// A unit vector pointing along the positive X axis.
    pub const X: Self = Self::new(T::ONE, T::ZERO);

    /// A unit vector pointing along the positive Y axis.
    pub const Y: Self = Self::new(T::ZERO, T::ONE);

    /// The unit axes.
    pub const AXES: [Self; 2] = [Self::X, Self::Y];
}

/// # Signed Constants
impl<T: Signed> GVec2<T> {
    /// All negative ones.
    pub const NEG_ONE: Self = Self::new(T::NEG_ONE, T::NEG_ONE);

    /// A unit vector pointing along the negative X axis.
    pub const NEG_X: Self = Self::new(T::NEG_ONE, T::ZERO);

    /// A unit vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self::new(T::ZERO, T::NEG_ONE);
}

/// # Float Constants
impl<T: Float> GVec2<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::new(T::NAN, T::NAN);

    /// All `INFINITY`.
    pub const INFINITY: Self = Self::new(T::INFINITY, T::INFINITY);

    /// All `NEG_INFINITY`.
    pub const NEG_INFINITY: Self = Self::new(T::NEG_INFINITY, T::NEG_INFINITY);
}

/// # Boolean Constants
impl<T: SimdBool> GVec2<T> {
    /// All `true`.
    pub const TRUE: Self = Self::new(T::TRUE, T::TRUE);

    /// All `false`.
    pub const FALSE: Self = Self::new(T::FALSE, T::FALSE);
}

/// # Constructors
impl<T: Copy> GVec2<T> {
    /// Creates a new vector.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Creates a new vector with all elements set to `v`.
    #[inline]
    #[must_use]
    pub const fn splat(v: T) -> Self {
        Self::new(v, v)
    }

    /// Returns a vector containing each element of `self` modified by a mapping function `f`.
    #[inline]
    #[must_use]
    pub fn map<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Self::new(f(self.x), f(self.y))
    }

    /// Creates a vector from the elements in `if_true` and `if_false`, selecting which to use
    /// based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select<B: Select<T>>(boolean: B, if_true: Self, if_false: Self) -> Self {
        Self {
            x: boolean.select(if_true.x, if_false.x),
            y: boolean.select(if_true.y, if_false.y),
        }
    }

    /// Creates a vector from the elements in `if_true` and `if_false`, selecting which to use
    /// based on the elements of `mask`.
    ///
    /// A true element in the mask uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select_mask<B: Select<T>>(mask: GVec2<B>, if_true: Self, if_false: Self) -> Self {
        Self {
            x: mask.x.select(if_true.x, if_false.x),
            y: mask.y.select(if_true.y, if_false.y),
        }
    }

    /// Creates a new vector from an array.
    #[inline]
    #[must_use]
    pub const fn from_array(arr: [T; 2]) -> Self {
        Self::new(arr[0], arr[1])
    }

    /// Returns the vector as an array.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [T; 2] {
        [self.x, self.y]
    }

    /// Creates a vector from the first 2 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 2 elements long.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() >= 2);
        Self::new(slice[0], slice[1])
    }

    /// Writes the elements of `self` to the first 2 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 2 elements long.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [T]) {
        slice[..2].copy_from_slice(&self.to_array());
    }

    /*
    /// Creates a 3D vector from `self` and the given `z` value.
    #[inline]
    #[must_use]
    pub const fn extend(self, z: T) -> GVec3<T> {
        GVec3::new(self.x, self.y, z)
    }
    */

    /// Creates a 2D vector from `self` with the given value of `x`.
    #[inline]
    #[must_use]
    pub const fn with_x(mut self, x: T) -> Self {
        self.x = x;
        self
    }

    /// Creates a 2D vector from `self` with the given value of `y`.
    #[inline]
    #[must_use]
    pub const fn with_y(mut self, y: T) -> Self {
        self.y = y;
        self
    }
}

/// # Element-Wise Equality
impl<T: Copy + NumEq> GVec2<T> {
    /// Returns a vector mask containing the result of a `==` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x == rhs.x, self.y == rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmpeq(self, rhs: Self) -> GVec2<T::Bool> {
        GVec2::new(self.x.num_eq(rhs.x), self.y.num_eq(rhs.y))
    }

    /// Returns a vector mask containing the result of a `!=` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x != rhs.x, self.y != rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmpne(self, rhs: Self) -> GVec2<T::Bool> {
        GVec2::new(self.x.num_ne(rhs.x), self.y.num_ne(rhs.y))
    }
}

/// # Element-Wise Ordering
impl<T: Copy + NumOrd> GVec2<T> {
    /// Returns a vector mask containing the result of a `>=` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x >= rhs.x, self.y >= rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmpge(self, rhs: Self) -> GVec2<T::Bool> {
        GVec2::new(self.x.num_ge(rhs.x), self.y.num_ge(rhs.y))
    }

    /// Returns a vector mask containing the result of a `>` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x > rhs.x, self.y > rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmpgt(self, rhs: Self) -> GVec2<T::Bool> {
        GVec2::new(self.x.num_gt(rhs.x), self.y.num_gt(rhs.y))
    }

    /// Returns a vector mask containing the result of a `<=` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x <= rhs.x, self.y <= rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmple(self, rhs: Self) -> GVec2<T::Bool> {
        GVec2::new(self.x.num_le(rhs.x), self.y.num_le(rhs.y))
    }

    /// Returns a vector mask containing the result of a `<` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x < rhs.x, self.y < rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmplt(self, rhs: Self) -> GVec2<T::Bool> {
        GVec2::new(self.x.num_lt(rhs.x), self.y.num_lt(rhs.y))
    }

    /// Returns a vector containing the minimum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[min(x, rhs.x), min(self.y, rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
    }

    /// Returns a vector containing the maximum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[max(self.x, rhs.x), max(self.y, rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
    }

    /// Element-wise clamping of values.
    ///
    /// Each element in `min` must be less-or-equal to the corresponding element in `max`.
    #[inline]
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Returns the horizontal minimum of `self`.
    ///
    /// In other words this computes `min(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn min_element(self) -> T {
        self.x.min(self.y)
    }

    /// Returns the horizontal maximum of `self`.
    ///
    /// In other words this computes `max(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn max_element(self) -> T {
        self.x.max(self.y)
    }
}

/// # Number Operations
impl<T: Num + Signed> GVec2<T> {
    /// Computes the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> T {
        (self.x * rhs.x) + (self.y * rhs.y)
    }

    /// Returns a vector where every element is the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self::splat(self.dot(rhs))
    }

    /// Returns the sum of all elements of `self`.
    ///
    /// In other words, this computes `self.x + self.y + ..`.
    #[inline]
    #[must_use]
    pub fn element_sum(self) -> T {
        self.x + self.y
    }

    /// Returns the product of all elements of `self`.
    ///
    /// In other words, this computes `self.x * self.y * ..`.
    #[inline]
    #[must_use]
    pub fn element_product(self) -> T {
        self.x * self.y
    }

    /// Computes the squared length of `self`.
    ///
    /// This is faster than [`length`](Self::length) as it avoids a square root operation.
    #[inline]
    #[must_use]
    #[doc(alias = "magnitude2")]
    pub fn length_squared(self) -> T {
        self.dot(self)
    }

    /// Compute the squared [Euclidean distance] between two points in space.
    ///
    /// [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
    #[inline]
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> T {
        (self - rhs).length_squared()
    }

    /// Returns a vector that is equal to `self` rotated by 90 degrees.
    #[inline]
    #[must_use]
    pub fn perp(self) -> Self {
        Self::new(-self.y, self.x)
    }

    /// The perpendicular dot product of `self` and `rhs`.
    /// Also known as the wedge product, 2D cross product, and determinant.
    #[doc(alias = "wedge")]
    #[doc(alias = "cross")]
    #[doc(alias = "determinant")]
    #[inline]
    #[must_use]
    pub fn perp_dot(self, rhs: Self) -> T {
        (self.x * rhs.y) - (self.y * rhs.x)
    }
}

impl<T: Copy + Signed> GVec2<T> {
    /// Returns a vector containing the absolute value of each element of `self`.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }

    /// Returns a vector with elements representing the sign of `self`.
    ///
    /// - `1.0` if the element is positive
    /// - `-1.0` if the element is negative
    /// - `NAN` if the element is `NAN` (only for [`Float`] types)
    #[inline]
    #[must_use]
    pub fn signum(self) -> Self {
        Self::new(self.x.signum(), self.y.signum())
    }
}

/// # Real Operations
impl<T: Real> GVec2<T> {
    /// Returns a vector with a length no less than `min` and no more than `max`.
    #[inline]
    #[must_use]
    pub fn clamp_length(self, min: T, max: T) -> Self {
        let length = self.length();
        let scale = (min / length).max(T::ONE).min(max / length);
        Self::new(self.x * scale, self.y * scale)
    }

    /// Returns a vector with a length no less than `min`.
    #[inline]
    #[must_use]
    pub fn clamp_length_min(self, min: T) -> Self {
        let length = self.length();
        let scale = (min / length).max(T::ONE);
        Self::new(self.x * scale, self.y * scale)
    }

    /// Returns a vector with a length no more than `max`.
    #[inline]
    #[must_use]
    pub fn clamp_length_max(self, max: T) -> Self {
        let length = self.length();
        let scale = (max / length).min(T::ONE);
        Self::new(self.x * scale, self.y * scale)
    }

    /// Returns a vector with signs of `rhs` and the magnitudes of `self`.
    #[inline]
    #[must_use]
    pub fn copysign(self, rhs: Self) -> Self {
        Self::new(self.x.copysign(rhs.x), self.y.copysign(rhs.y))
    }

    /// Computes the length of `self`.
    #[inline]
    #[must_use]
    #[doc(alias = "magnitude")]
    pub fn length(self) -> T {
        self.length_squared().sqrt()
    }

    /// Computes `1.0 / length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn length_recip(self) -> T {
        T::ONE / self.length()
    }

    /// Computes the [Euclidean distance] between two points in space.
    ///
    /// [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
    #[inline]
    #[must_use]
    pub fn distance(self, rhs: Self) -> T {
        (self - rhs).length()
    }

    /// Returns `self` normalized to length `1.0`.
    ///
    /// For valid results, `self` must be finite and _not_ of length zero, nor very close to zero.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        self.mul(self.length_recip())
    }

    /// Returns whether `self` is length `1.0` or not, within a certain `eps` tolerance.
    #[inline]
    #[must_use]
    pub fn is_normalized(self, eps: T) -> T::Bool {
        (self.length_squared() + T::NEG_ONE).abs().num_le(eps)
    }

    /// Returns the vector projection of `self` onto `rhs`.
    ///
    /// `rhs` must be of non-zero length.
    #[inline]
    #[must_use]
    pub fn project_onto(self, rhs: Self) -> Self {
        rhs * self.dot(rhs) / rhs.length_squared()
    }

    /// Returns the vector rejection of `self` from `rhs`.
    ///
    /// The vector rejection is the vector perpendicular to the projection of `self` onto
    /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
    ///
    /// `rhs` must be of non-zero length.
    #[inline]
    #[must_use]
    pub fn reject_from(self, rhs: Self) -> Self {
        self - self.project_onto(rhs)
    }

    /// Returns the vector projection of `self` onto `rhs`.
    ///
    /// `rhs` must be normalized.
    #[inline]
    #[must_use]
    pub fn project_onto_normalized(self, rhs: Self) -> Self {
        rhs * self.dot(rhs)
    }

    /// Returns the vector rejection of `self` from `rhs`.
    ///
    /// The vector rejection is the vector perpendicular to the projection of `self` onto
    /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
    ///
    /// `rhs` must be normalized.
    #[inline]
    #[must_use]
    pub fn reject_from_normalized(self, rhs: Self) -> Self {
        self - self.project_onto_normalized(rhs)
    }

    /// Returns a vector containing the nearest integer to a number for each element of `self`.
    /// Round half-way cases away from zero.
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round())
    }

    /// Returns a vector containing the largest integer less than or equal to a number for each
    /// element of `self`.
    #[inline]
    #[must_use]
    pub fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor())
    }

    /// Returns a vector containing the smallest integer greater than or equal to a number for
    /// each element of `self`.
    #[inline]
    #[must_use]
    pub fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil())
    }

    /// Returns a vector containing the integer part each element of `self`. This means numbers are
    /// always truncated towards zero.
    #[inline]
    #[must_use]
    pub fn trunc(self) -> Self {
        Self::new(self.x.trunc(), self.y.trunc())
    }

    /// Returns a vector containing `0.0` if `rhs < self` and `1.0` otherwise.
    ///
    /// Similar to GLSL's `step(edge, x)`, which translates into `edge.step(x)`.
    #[inline]
    #[must_use]
    pub fn step(self, rhs: Self) -> Self {
        Self::select_mask(rhs.cmplt(self), Self::ZERO, Self::ONE)
    }

    /// Returns a vector containing all elements of `self` clamped to the range of `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn saturate(self) -> Self {
        self.clamp(Self::ZERO, Self::ONE)
    }

    /// Returns a vector containing the fractional part of the vector as `self - self.trunc()`.
    ///
    /// Note that this differs from the GLSL implementation of `fract`, which returns
    /// `self - self.floor()`.
    ///
    /// Note that this is fast but not precise for large numbers.
    #[inline]
    #[must_use]
    pub fn fract(self) -> Self {
        self - self.trunc()
    }

    /// Returns a vector containing the fractional part of the vector as `self - self.floor()`.
    ///
    /// Note that this differs from the Rust implementation of `fract`, which returns
    /// `self - self.trunc()`.
    ///
    /// Note that this is fast but not precise for large numbers.
    #[inline]
    #[must_use]
    pub fn fract_gl(self) -> Self {
        self - self.floor()
    }

    /// Returns a vector containing `e^self` (the exponential function) for each element of
    /// `self`.
    #[inline]
    #[must_use]
    pub fn exp(self) -> Self {
        Self::new(self.x.exp(), self.y.exp())
    }

    /// Returns a vector containing `2^self` for each element of `self`.
    #[inline]
    #[must_use]
    pub fn exp2(self) -> Self {
        Self::new(self.x.exp2(), self.y.exp2())
    }

    /// Returns a vector containing the natural logarithm for each element of `self`.
    /// This returns NaN when the element is negative and negative infinity when the element is zero.
    #[inline]
    #[must_use]
    pub fn ln(self) -> Self {
        Self::new(self.x.ln(), self.y.ln())
    }

    /// Returns a vector containing the base 2 logarithm for each element of `self`.
    /// This returns NaN when the element is negative and negative infinity when the element is zero.
    #[inline]
    #[must_use]
    pub fn log2(self) -> Self {
        Self::new(self.x.log2(), self.y.log2())
    }

    /// Returns a vector containing the square root for each element of `self`.
    /// This returns NaN when the element is negative.
    #[inline]
    #[must_use]
    pub fn sqrt(self) -> Self {
        Self::new(self.x.sqrt(), self.y.sqrt())
    }

    /// Returns a vector containing the cosine for each element of `self`.
    #[inline]
    #[must_use]
    pub fn cos(self) -> Self {
        Self::new(self.x.cos(), self.y.cos())
    }

    /// Returns a vector containing the sine for each element of `self`.
    #[inline]
    #[must_use]
    pub fn sin(self) -> Self {
        Self::new(self.x.sin(), self.y.sin())
    }

    /// Returns a tuple of two vectors containing the sine and cosine for each element of `self`.
    #[inline]
    #[must_use]
    pub fn sin_cos(self) -> (Self, Self) {
        let (sin_x, cos_x) = self.x.sin_cos();
        let (sin_y, cos_y) = self.y.sin_cos();

        (Self::new(sin_x, sin_y), Self::new(cos_x, cos_y))
    }

    /// Returns a vector containing the reciprocal `1.0 / n` of each element of `self`.
    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        Self::new(T::ONE / self.x, T::ONE / self.y)
    }

    /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
    /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
    /// extrapolated.
    #[inline]
    #[must_use]
    pub fn lerp(self, rhs: Self, s: T) -> Self {
        self * (T::ONE - s) + rhs * s
    }

    /// Calculates the midpoint between `self` and `rhs`.
    ///
    /// The midpoint is the average of, or halfway point between, two vectors.
    /// `a.midpoint(b)` should yield the same result as `a.lerp(b, 0.5)`
    /// while being slightly cheaper to compute.
    #[inline]
    pub fn midpoint(self, rhs: Self) -> Self {
        Self::new(self.x.midpoint(rhs.x), self.y.midpoint(rhs.y))
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs` is
    /// less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two vectors contain similar elements. It works best when
    /// comparing with a known value. The `max_abs_diff` that should be used depends on
    /// the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: T) -> T::Bool {
        self.sub(rhs).abs().cmple(Self::splat(max_abs_diff)).all()
    }

    /// Returns the reflection vector for a given incident vector `self` and surface normal
    /// `normal`.
    ///
    /// `normal` must be normalized.
    #[inline]
    #[must_use]
    pub fn reflect(self, normal: Self) -> Self {
        // TODO: A nicer way to just use number literals would be neat
        self - normal * ((T::ONE + T::ONE) * self.dot(normal))
    }

    /// Returns the refraction direction for a given incident vector `self`, surface normal
    /// `normal` and ratio of indices of refraction, `eta`. When total internal reflection occurs,
    /// a zero vector will be returned.
    ///
    /// `self` and `normal` must be normalized.
    #[inline]
    pub fn refract(self, normal: Self, eta: T) -> Self {
        let n = normal;
        let one = T::ONE;
        let ndi = n.dot(self);

        let k = one - eta * eta * (one - ndi * ndi);
        let mask = k.num_gt(T::ZERO);
        let out = self * eta - n * (eta * ndi + k.sqrt());

        Self::select(mask, Self::ZERO, out)
    }

    /// Creates a 2D vector containing `[angle.cos(), angle.sin()]`.
    ///
    /// This can be used in conjunction with the [`rotate`][Self::rotate()] method,
    /// for example `GVec2::from_angle(PI).rotate(GVec2::Y)` will create the vector `[-1, 0]`
    /// and rotate [`GVec2::Y`] around it, returning `-GVec2::Y`.
    #[inline]
    #[must_use]
    pub fn from_angle(angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(cos, sin)
    }

    /// Returns the angle (in radians) of this vector in the range `[-π, +π]`.
    ///
    /// The input does not need to be a unit vector however it must be non-zero.
    #[inline]
    #[must_use]
    pub fn to_angle(self) -> T {
        self.y.atan2(self.x)
    }

    /// Returns the angle of rotation (in radians) from `self` to `rhs` in the range `[-π, +π]`.
    ///
    /// The inputs do not need to be unit vectors however they must be non-zero.
    #[inline]
    #[must_use]
    pub fn angle_to(self, rhs: Self) -> T {
        let dot = self.dot(rhs) / (self.length_squared() * rhs.length_squared()).sqrt();
        // TODO: Glam uses a custom acos approximation here. It would probably be faster?
        let angle = dot.clamp(T::NEG_ONE, T::ONE).acos();

        angle * self.perp_dot(rhs).signum()
    }

    /// Returns `rhs` rotated by the angle of `self`.
    ///
    /// If `self` is normalized, then this just rotation.
    /// This is what you usually want. Otherwise, it will be
    /// like a rotation with a multiplication by `self`'s length.
    ///
    /// This can be used in conjunction with the [`from_angle`][Self::from_angle()] method,
    /// for example, `GVec2::from_angle(PI).rotate(GVec2::Y)` will create the vector `[-1, 0]`
    /// and rotate [`GVec2::Y`] around it, returning `-GVec2::Y`.
    #[inline]
    #[must_use]
    pub fn rotate(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x - self.y * rhs.y,
            y: self.y * rhs.x + self.x * rhs.y,
        }
    }

    /// Rotates towards `rhs` up to `max_angle` (in radians).
    ///
    /// When `max_angle` is `T::ZERO`, the result will be equal to `self`. When `max_angle` is equal to
    /// `self.angle_between(rhs)`, the result will be parallel to `rhs`. If `max_angle` is negative,
    /// rotates towards the exact opposite of `rhs`. Will not go past the target.
    #[inline]
    #[must_use]
    pub fn rotate_towards(self, rhs: Self, max_angle: T) -> Self {
        let a = self.angle_to(rhs);
        let abs_a = a.abs();
        // When `max_angle < 0`, rotate no further than `PI` radians away
        let angle = max_angle.clamp(abs_a - T::PI, abs_a) * a.signum();
        Self::from_angle(angle).rotate(self)
    }
}

/// # Float Methods
impl<T: Float> GVec2<T> {
    /// Returns true if, and only if, all elements are finite.  If any element is either
    /// `NaN`, positive or negative infinity, this will return false.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> T::Bool {
        self.x.is_finite() & self.y.is_finite()
    }

    /// Performs [`is_finite`](Float::is_finite) on each element of self, returning a vector mask of the results.
    ///
    /// In other words, this computes `[x.is_finite(), y.is_finite(), ...]`.
    #[inline]
    #[must_use]
    pub fn is_finite_mask(self) -> GVec2<T::Bool> {
        GVec2::new(self.x.is_finite(), self.y.is_finite())
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> T::Bool {
        self.x.is_nan() | self.y.is_nan()
    }

    /// Performs [`is_nan`](Float::is_nan) on each element of self, returning a vector mask of the results.
    ///
    /// In other words, this computes `[x.is_nan(), y.is_nan(), ...]`.
    #[inline]
    #[must_use]
    pub fn is_nan_mask(self) -> GVec2<T::Bool> {
        GVec2::new(self.x.is_nan(), self.y.is_nan())
    }

    /// Returns `self` normalized to length `1.0` if possible, else returns a fallback value.
    ///
    /// In particular, if the input is zero (or very close to zero), or non-finite,
    /// the result of this operation will be the fallback value.
    #[inline]
    #[must_use]
    pub fn normalize_or(self, fallback: Self) -> Self {
        let rcp = self.length_recip();
        let mask = rcp.is_finite() & rcp.num_gt(T::ZERO);
        GVec2::select(mask, self * rcp, fallback)
    }

    /// Returns `self` normalized to length `1.0` if possible, else returns zero.
    ///
    /// In particular, if the input is zero (or very close to zero), or non-finite,
    /// the result of this operation will be zero.
    #[inline]
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        self.normalize_or(Self::ZERO)
    }

    /// Returns `self` normalized to length `1.0` and the length of `self`.
    ///
    /// If `self` is zero length, then `(Self::X, T::ZERO)` is returned.
    #[inline]
    #[must_use]
    pub fn normalize_and_length(self) -> (Self, T) {
        let length = self.length();
        let rcp = T::ONE / length;
        let mask = rcp.is_finite() & rcp.num_gt(T::ZERO);
        (
            Self::select(mask, self * rcp, Self::X),
            mask.select(length, T::ZERO),
        )
    }

    /// Returns a vector containing each element of `self` raised to the power of `n`.
    #[inline]
    #[must_use]
    pub fn powf(self, n: T) -> Self {
        Self::new(self.x.powf(n), self.y.powf(n))
    }

    /// Moves towards `rhs` based on the value `d`.
    ///
    /// When `d` is `0.0`, the result will be equal to `self`. When `d` is equal to
    /// `self.distance(rhs)`, the result will be equal to `rhs`. Will not go past `rhs`.
    #[inline]
    #[must_use]
    pub fn move_towards(self, rhs: Self, d: T) -> Self {
        let a = rhs - self;
        let len = a.length();
        let result = self + a / len * d;
        let mask = len.num_le(d) | len.num_le(T::EPSILON);
        Self::select(mask, rhs, result)
    }

    /// Fused multiply-add. Computes `(self * a) + b` element-wise with only one rounding
    /// error, yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if the target
    /// architecture has a dedicated fma CPU instruction. However, this is not always true,
    /// and will be heavily dependant on designing algorithms with specific target hardware in
    /// mind.
    #[inline]
    #[must_use]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new(self.x.mul_add(a.x, b.x), self.y.mul_add(a.y, b.y))
    }
}

/// # Integer Methods
impl<T: Int + Signed> GVec2<T>
where
    <T as Signed>::Unsigned: Int,
{
    /// Computes the [Manhattan distance] between two points in space.
    ///
    /// [Manhattan distance]: https://en.wikipedia.org/wiki/Taxicab_geometry
    #[inline]
    #[must_use]
    pub fn manhattan_distance(self, rhs: Self) -> <T as Signed>::Unsigned {
        self.x.abs_diff(rhs.x) + self.y.abs_diff(rhs.y)
    }

    /// Computes the [Chebyshev distance] between two points in space.
    ///
    /// [Chebyshev distance]: https://en.wikipedia.org/wiki/Chebyshev_distance
    #[inline]
    #[must_use]
    pub fn chebyshev_distance(self, rhs: Self) -> <T as Signed>::Unsigned {
        self.x.abs_diff(rhs.x).max(self.y.abs_diff(rhs.y))
    }
}

/// # Checked Operations
impl<T: Copy + CheckedAdd<Output = T>> GVec2<T> {
    /// Returns a vector containing the checked addition of `self` and `rhs`.
    ///
    /// In other words, this computes `Some([self.x + rhs.x, self.y + rhs.y, ..])` but returns `None` on any overflow.
    #[inline]
    #[must_use]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(Self::new(
            self.x.checked_add(rhs.x)?,
            self.y.checked_add(rhs.y)?,
        ))
    }
}

impl<T: Copy + CheckedSub<Output = T>> GVec2<T> {
    /// Returns a vector containing the checked subtraction of `self` and `rhs`.
    ///
    /// In other words, this computes `Some([self.x - rhs.x, self.y - rhs.y, ..])` but returns `None` on any overflow.
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        Some(Self::new(
            self.x.checked_sub(rhs.x)?,
            self.y.checked_sub(rhs.y)?,
        ))
    }
}

impl<T: Copy + CheckedMul<Output = T>> GVec2<T> {
    /// Returns a vector containing the checked multiplication of `self` and `rhs`.
    ///
    /// In other words, this computes `Some([self.x * rhs.x, self.y * rhs.y, ..])` but returns `None` on any overflow.
    #[inline]
    #[must_use]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        Some(Self::new(
            self.x.checked_mul(rhs.x)?,
            self.y.checked_mul(rhs.y)?,
        ))
    }
}

impl<T: Copy + CheckedDiv<Output = T>> GVec2<T> {
    /// Returns a vector containing the checked division of `self` and `rhs`.
    ///
    /// In other words, this computes `Some([self.x / rhs.x, self.y / rhs.y, ..])` but returns `None` on any division by zero or overflow.
    #[inline]
    #[must_use]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        Some(Self::new(
            self.x.checked_div(rhs.x)?,
            self.y.checked_div(rhs.y)?,
        ))
    }
}

/// # Wrapping Operations
impl<T: Copy + WrappingAdd<Output = T>> GVec2<T> {
    /// Returns a vector containing the wrapping addition of `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x.wrapping_add(rhs.x), self.y.wrapping_add(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn wrapping_add(self, rhs: Self) -> Self {
        Self::new(self.x.wrapping_add(rhs.x), self.y.wrapping_add(rhs.y))
    }
}

impl<T: Copy + WrappingSub<Output = T>> GVec2<T> {
    /// Returns a vector containing the wrapping subtraction of `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x.wrapping_sub(rhs.x), self.y.wrapping_sub(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        Self::new(self.x.wrapping_sub(rhs.x), self.y.wrapping_sub(rhs.y))
    }
}

impl<T: Copy + WrappingMul<Output = T>> GVec2<T> {
    /// Returns a vector containing the wrapping multiplication of `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x.wrapping_mul(rhs.x), self.y.wrapping_mul(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        Self::new(self.x.wrapping_mul(rhs.x), self.y.wrapping_mul(rhs.y))
    }
}

impl<T: Copy + WrappingDiv<Output = T>> GVec2<T> {
    /// Returns a vector containing the wrapping division of `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x.wrapping_div(rhs.x), self.y.wrapping_div(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn wrapping_div(self, rhs: Self) -> Self {
        Self::new(self.x.wrapping_div(rhs.x), self.y.wrapping_div(rhs.y))
    }
}

/// # Saturating Operations
impl<T: Copy + SaturatingAdd<Output = T>> GVec2<T> {
    /// Returns a vector containing the saturating addition of `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x.saturating_add(rhs.x), self.y.saturating_add(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self::new(self.x.saturating_add(rhs.x), self.y.saturating_add(rhs.y))
    }
}

impl<T: Copy + SaturatingSub<Output = T>> GVec2<T> {
    /// Returns a vector containing the saturating subtraction of `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x.saturating_sub(rhs.x), self.y.saturating_sub(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self::new(self.x.saturating_sub(rhs.x), self.y.saturating_sub(rhs.y))
    }
}

impl<T: Copy + SaturatingMul<Output = T>> GVec2<T> {
    /// Returns a vector containing the saturating multiplication of `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x.saturating_mul(rhs.x), self.y.saturating_mul(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        Self::new(self.x.saturating_mul(rhs.x), self.y.saturating_mul(rhs.y))
    }
}

impl<T: Copy + SaturatingDiv<Output = T>> GVec2<T> {
    /// Returns a vector containing the saturating division of `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x.saturating_div(rhs.x), self.y.saturating_div(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn saturating_div(self, rhs: Self) -> Self {
        Self::new(self.x.saturating_div(rhs.x), self.y.saturating_div(rhs.y))
    }
}

/// # Boolean Operations
impl<T: SimdBool> GVec2<T> {
    /// Returns `true` if all elements of `self` are true, and `false` otherwise.
    #[inline]
    #[must_use]
    pub fn all(self) -> T {
        self.x & self.y
    }

    /// Returns `true` if any element of `self` is true, and `false` otherwise.
    #[inline]
    #[must_use]
    pub fn any(self) -> T {
        self.x | self.y
    }

    /// Tests the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    #[must_use]
    pub fn test(self, index: usize) -> T {
        match index {
            0 => self.x,
            1 => self.y,
            _ => panic!("index out of bounds"),
        }
    }

    /// Sets the element at `index` to `value`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        match index {
            0 => self.x = value,
            1 => self.y = value,
            _ => panic!("index out of bounds"),
        }
    }
}

impl<T: Copy + Add<Output = T>> Add for GVec2<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: GVec2<T>) -> Self {
        GVec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<T: Copy + Add<Output = T>> Add<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &GVec2<T>) -> Self {
        self.add(*rhs)
    }
}

impl<T: Copy + Add<Output = T>> Add<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn add(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).add(rhs)
    }
}

impl<T: Copy + Add<Output = T>> Add<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn add(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).add(*rhs)
    }
}

impl<T: Copy + AddAssign> AddAssign for GVec2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: GVec2<T>) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: Copy + AddAssign> AddAssign<&GVec2<T>> for GVec2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &GVec2<T>) {
        self.add_assign(*rhs);
    }
}

impl<T: Copy + Sub<Output = T>> Sub for GVec2<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: GVec2<T>) -> Self {
        GVec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<T: Copy + Sub<Output = T>> Sub<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &GVec2<T>) -> Self {
        self.sub(*rhs)
    }
}

impl<T: Copy + Sub<Output = T>> Sub<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn sub(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).sub(rhs)
    }
}

impl<T: Copy + Sub<Output = T>> Sub<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn sub(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Copy + SubAssign> SubAssign for GVec2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: GVec2<T>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: Copy + SubAssign> SubAssign<&GVec2<T>> for GVec2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &GVec2<T>) {
        self.sub_assign(*rhs);
    }
}

impl<T: Copy + Mul<Output = T>> Mul for GVec2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: GVec2<T>) -> Self {
        GVec2::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl<T: Copy + Mul<Output = T>> Mul<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &GVec2<T>) -> Self {
        self.mul(*rhs)
    }
}

impl<T: Copy + Mul<Output = T>> Mul<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Copy + Mul<Output = T>> Mul<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Copy + Mul<Output = T>> Mul<T> for GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: T) -> GVec2<T> {
        GVec2::new(self.x * rhs, self.y * rhs)
    }
}

impl<T: Copy + Mul<Output = T>> Mul<&T> for GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GVec2<T> {
        self.mul(*rhs)
    }
}

impl<T: Copy + Mul<Output = T>> Mul<T> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: T) -> GVec2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Copy + Mul<Output = T>> Mul<&T> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GVec2<T> {
        (*self).mul(*rhs)
    }
}

// We cannot implement scalar * vector generically because of Rust's orphan rules.
macro_rules! impl_scalar_left_mul {
    ($($t:ty),*) => {
        $(
            impl Mul<GVec2<$t>> for $t {
                type Output = GVec2<$t>;
                #[inline]
                fn mul(self, rhs: GVec2<$t>) -> GVec2<$t> {
                    GVec2::new(self * rhs.x, self * rhs.y)
                }
            }

            impl Mul<&GVec2<$t>> for $t {
                type Output = GVec2<$t>;
                #[inline]
                fn mul(self, rhs: &GVec2<$t>) -> GVec2<$t> {
                    self.mul(*rhs)
                }
            }

            impl Mul<GVec2<$t>> for &$t {
                type Output = GVec2<$t>;
                #[inline]
                fn mul(self, rhs: GVec2<$t>) -> GVec2<$t> {
                    (*self).mul(rhs)
                }
            }

            impl Mul<&GVec2<$t>> for &$t {
                type Output = GVec2<$t>;
                #[inline]
                fn mul(self, rhs: &GVec2<$t>) -> GVec2<$t> {
                    (*self).mul(*rhs)
                }
            }
        )*
    };
}

// TODO: Implement for SIMD types
impl_scalar_left_mul!(i8, i16, i32, i64, i128, isize);
impl_scalar_left_mul!(u8, u16, u32, u64, u128, usize);
impl_scalar_left_mul!(f32, f64);

impl<T: Copy + MulAssign> MulAssign for GVec2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GVec2<T>) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl<T: Copy + MulAssign> MulAssign<&GVec2<T>> for GVec2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GVec2<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Copy + MulAssign> MulAssign<T> for GVec2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: Copy + MulAssign> MulAssign<&T> for GVec2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.mul_assign(*rhs);
    }
}

impl<T: Copy + Div<Output = T>> Div for GVec2<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: GVec2<T>) -> Self {
        GVec2::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl<T: Copy + Div<Output = T>> Div<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: &GVec2<T>) -> Self {
        self.div(*rhs)
    }
}

impl<T: Copy + Div<Output = T>> Div<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn div(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).div(rhs)
    }
}

impl<T: Copy + Div<Output = T>> Div<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn div(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).div(*rhs)
    }
}

impl<T: Copy + Div<Output = T>> Div<T> for GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn div(self, rhs: T) -> GVec2<T> {
        GVec2::new(self.x / rhs, self.y / rhs)
    }
}

impl<T: Copy + Div<Output = T>> Div<&T> for GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn div(self, rhs: &T) -> GVec2<T> {
        self.div(*rhs)
    }
}

impl<T: Copy + Div<Output = T>> Div<T> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn div(self, rhs: T) -> GVec2<T> {
        (*self).div(rhs)
    }
}

impl<T: Copy + Div<Output = T>> Div<&T> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn div(self, rhs: &T) -> GVec2<T> {
        (*self).div(*rhs)
    }
}

impl<T: Copy + DivAssign> DivAssign for GVec2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: GVec2<T>) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}

impl<T: Copy + DivAssign> DivAssign<&GVec2<T>> for GVec2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &GVec2<T>) {
        self.div_assign(*rhs);
    }
}

impl<T: Copy + DivAssign> DivAssign<T> for GVec2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T: Copy + DivAssign> DivAssign<&T> for GVec2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.div_assign(*rhs);
    }
}

impl<T: Copy + DivEuclid<Output = T>> DivEuclid for GVec2<T> {
    type Output = Self;
    #[inline]
    fn div_euclid(self, rhs: Self) -> Self {
        Self::new(self.x.div_euclid(rhs.x), self.y.div_euclid(rhs.y))
    }
}

impl<T: Copy + DivEuclid<Output = T>> DivEuclid<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn div_euclid(self, rhs: &GVec2<T>) -> Self {
        self.div_euclid(*rhs)
    }
}

impl<T: Copy + DivEuclid<Output = T>> DivEuclid<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn div_euclid(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).div_euclid(rhs)
    }
}

impl<T: Copy + DivEuclid<Output = T>> DivEuclid<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn div_euclid(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).div_euclid(*rhs)
    }
}

impl<T: Copy + Rem<Output = T>> Rem for GVec2<T> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: GVec2<T>) -> Self {
        GVec2::new(self.x % rhs.x, self.y % rhs.y)
    }
}

impl<T: Copy + Rem<Output = T>> Rem<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: &GVec2<T>) -> Self {
        self.rem(*rhs)
    }
}

impl<T: Copy + Rem<Output = T>> Rem<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn rem(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).rem(rhs)
    }
}

impl<T: Copy + Rem<Output = T>> Rem<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn rem(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).rem(*rhs)
    }
}

impl<T: Copy + RemAssign> RemAssign for GVec2<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: GVec2<T>) {
        self.x %= rhs.x;
        self.y %= rhs.y;
    }
}

impl<T: Copy + RemAssign> RemAssign<&GVec2<T>> for GVec2<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: &GVec2<T>) {
        self.rem_assign(*rhs);
    }
}

impl<T: Copy + RemEuclid<Output = T>> RemEuclid for GVec2<T> {
    type Output = Self;
    #[inline]
    fn rem_euclid(self, rhs: Self) -> Self {
        Self::new(self.x.rem_euclid(rhs.x), self.y.rem_euclid(rhs.y))
    }
}

impl<T: Copy + RemEuclid<Output = T>> RemEuclid<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn rem_euclid(self, rhs: &GVec2<T>) -> Self {
        self.rem_euclid(*rhs)
    }
}

impl<T: Copy + RemEuclid<Output = T>> RemEuclid<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn rem_euclid(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).rem_euclid(rhs)
    }
}

impl<T: Copy + RemEuclid<Output = T>> RemEuclid<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn rem_euclid(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).rem_euclid(*rhs)
    }
}

impl<T: Copy + Neg<Output = T>> Neg for GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn neg(self) -> GVec2<T> {
        GVec2::new(-self.x, -self.y)
    }
}

impl<T: Copy + Neg<Output = T>> Neg for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn neg(self) -> GVec2<T> {
        (*self).neg()
    }
}

impl<T: Copy + BitAnd<Output = T>> BitAnd for GVec2<T> {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        GVec2::new(self.x & rhs.x, self.y & rhs.y)
    }
}

impl<T: Copy + BitAnd<Output = T>> BitAnd<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: &GVec2<T>) -> Self {
        self.bitand(*rhs)
    }
}

impl<T: Copy + BitAnd<Output = T>> BitAnd<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn bitand(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).bitand(rhs)
    }
}

impl<T: Copy + BitAnd<Output = T>> BitAnd<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn bitand(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).bitand(*rhs)
    }
}

impl<T: Copy + BitAndAssign> BitAndAssign for GVec2<T> {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.x &= rhs.x;
        self.y &= rhs.y;
    }
}

impl<T: Copy + BitAndAssign> BitAndAssign<&GVec2<T>> for GVec2<T> {
    #[inline]
    fn bitand_assign(&mut self, rhs: &Self) {
        self.bitand_assign(*rhs);
    }
}

impl<T: Copy + BitOr<Output = T>> BitOr for GVec2<T> {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        GVec2::new(self.x | rhs.x, self.y | rhs.y)
    }
}

impl<T: Copy + BitOr<Output = T>> BitOr<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: &GVec2<T>) -> Self {
        self.bitor(*rhs)
    }
}

impl<T: Copy + BitOr<Output = T>> BitOr<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn bitor(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).bitor(rhs)
    }
}

impl<T: Copy + BitOr<Output = T>> BitOr<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn bitor(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).bitor(*rhs)
    }
}

impl<T: Copy + BitOrAssign> BitOrAssign for GVec2<T> {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.x |= rhs.x;
        self.y |= rhs.y;
    }
}

impl<T: Copy + BitOrAssign> BitOrAssign<&GVec2<T>> for GVec2<T> {
    #[inline]
    fn bitor_assign(&mut self, rhs: &Self) {
        self.bitor_assign(*rhs);
    }
}

impl<T: Copy + BitXor<Output = T>> BitXor for GVec2<T> {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        GVec2::new(self.x ^ rhs.x, self.y ^ rhs.y)
    }
}

impl<T: Copy + BitXor<Output = T>> BitXor<&GVec2<T>> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: &GVec2<T>) -> Self {
        self.bitxor(*rhs)
    }
}

impl<T: Copy + BitXor<Output = T>> BitXor<GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn bitxor(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).bitxor(rhs)
    }
}

impl<T: Copy + BitXor<Output = T>> BitXor<&GVec2<T>> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn bitxor(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).bitxor(*rhs)
    }
}

impl<T: Copy + BitXorAssign> BitXorAssign for GVec2<T> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.x ^= rhs.x;
        self.y ^= rhs.y;
    }
}

impl<T: Copy + BitXorAssign> BitXorAssign<&GVec2<T>> for GVec2<T> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.bitxor_assign(*rhs);
    }
}

impl<T: Copy + Not<Output = T>> Not for GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn not(self) -> GVec2<T> {
        GVec2::new(!self.x, !self.y)
    }
}

impl<T: Copy + Not<Output = T>> Not for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn not(self) -> GVec2<T> {
        (*self).not()
    }
}

impl<T: Int> Shl<T> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn shl(self, rhs: T) -> Self {
        GVec2::new(self.x << rhs, self.y << rhs)
    }
}

impl<T: Int> Shl<&T> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn shl(self, rhs: &T) -> Self {
        self.shl(*rhs)
    }
}

impl<T: Int> Shl<T> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn shl(self, rhs: T) -> GVec2<T> {
        (*self).shl(rhs)
    }
}

impl<T: Int> Shl<&T> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn shl(self, rhs: &T) -> GVec2<T> {
        (*self).shl(*rhs)
    }
}

impl<T: Int> ShlAssign<T> for GVec2<T> {
    #[inline]
    fn shl_assign(&mut self, rhs: T) {
        self.x <<= rhs;
        self.y <<= rhs;
    }
}

impl<T: Int> ShlAssign<&T> for GVec2<T> {
    #[inline]
    fn shl_assign(&mut self, rhs: &T) {
        self.shl_assign(*rhs);
    }
}

impl<T: Int> Shr<T> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn shr(self, rhs: T) -> Self {
        GVec2::new(self.x >> rhs, self.y >> rhs)
    }
}

impl<T: Int> Shr<&T> for GVec2<T> {
    type Output = Self;
    #[inline]
    fn shr(self, rhs: &T) -> Self {
        self.shr(*rhs)
    }
}

impl<T: Int> Shr<T> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn shr(self, rhs: T) -> GVec2<T> {
        (*self).shr(rhs)
    }
}

impl<T: Int> Shr<&T> for &GVec2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn shr(self, rhs: &T) -> GVec2<T> {
        (*self).shr(*rhs)
    }
}

impl<T: Int> ShrAssign<T> for GVec2<T> {
    #[inline]
    fn shr_assign(&mut self, rhs: T) {
        self.x >>= rhs;
        self.y >>= rhs;
    }
}

impl<T: Int> ShrAssign<&T> for GVec2<T> {
    #[inline]
    fn shr_assign(&mut self, rhs: &T) {
        self.shr_assign(*rhs);
    }
}

impl<T: Num> Sum<GVec2<T>> for GVec2<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, T: Num> Sum<&'a GVec2<T>> for GVec2<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl<T: Num> Product<GVec2<T>> for GVec2<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl<'a, T: Num> Product<&'a GVec2<T>> for GVec2<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ONE, |a, &b| a * b)
    }
}

impl<T: Copy> Index<usize> for GVec2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            i => panic!(
                "index {i} out of bounds for vector of type: {}",
                core::any::type_name::<GVec2<T>>()
            ),
        }
    }
}

impl<T: Copy> IndexMut<usize> for GVec2<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            i => panic!(
                "index {i} out of bounds for vector of type: {}",
                core::any::type_name::<GVec2<T>>()
            ),
        }
    }
}

/*impl<T: Copy> From<GVec3<T>> for GVec2<T> {
    #[inline]
    fn from(vec: GVec3<T>) -> Self {
        Self { x: vec.x, y: vec.y }
    }
}*/

impl<T: Copy> From<[T; 2]> for GVec2<T> {
    #[inline]
    fn from(comps: [T; 2]) -> Self {
        Self::new(comps[0], comps[1])
    }
}

impl<T: Copy> From<GVec2<T>> for [T; 2] {
    #[inline]
    fn from(v: GVec2<T>) -> Self {
        [v.x, v.y]
    }
}

impl<T: Copy> From<(T, T)> for GVec2<T> {
    #[inline]
    fn from(comps: (T, T)) -> Self {
        Self::new(comps.0, comps.1)
    }
}

impl<T: Copy> From<GVec2<T>> for (T, T) {
    #[inline]
    fn from(v: GVec2<T>) -> Self {
        (v.x, v.y)
    }
}

impl<T: Copy> AsRef<[T; 2]> for GVec2<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 2] {
        unsafe { &*(self as *const GVec2<T> as *const [T; 2]) }
    }
}

impl<T: Copy> AsMut<[T; 2]> for GVec2<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 2] {
        unsafe { &mut *(self as *mut GVec2<T> as *mut [T; 2]) }
    }
}

impl<T: Copy + core::fmt::Display> core::fmt::Display for GVec2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{}, {}]", self.x, self.y)
    }
}

impl<T: Copy + core::fmt::Debug> core::fmt::Debug for GVec2<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_tuple(stringify!(GVec2<T>))
            .field(&self.x)
            .field(&self.y)
            .finish()
    }
}
