use crate::rotation::Rot3A;
use crate::vector::{BVec3A, GVec2, GVec3, GVec4};

use core::{
    fmt,
    iter::{Product, Sum},
    ops::*,
};

use gimd::f32x4;
use gnum::{
    cmp::{NumEq, NumOrd},
    num::{
        Float, NumCast, Real, Signed,
        ops::{DivEuclid, RemEuclid},
    },
    simd::{Select, Shuffle4, SimdLike},
};

/// Creates a 3-dimensional vector with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Vec3`](crate::vector::Vec3),
/// at the cost of a larger size and slightly more expensive construction
/// and field access.
#[inline(always)]
#[must_use]
pub const fn vec3a(x: f32, y: f32, z: f32) -> Vec3A {
    Vec3A::new(x, y, z)
}

/// A 3-dimensional vector with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Vec3`](crate::vector::Vec3)
/// at the cost of a larger size and slightly more expensive construction
/// and field access.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[cfg_attr(
    feature = "zerocopy",
    derive(
        zerocopy_derive::FromBytes,
        zerocopy_derive::Immutable,
        zerocopy_derive::IntoBytes,
        zerocopy_derive::KnownLayout
    )
)]
#[repr(transparent)]
pub struct Vec3A(f32x4);

/// Wraps a SIMD comparison mask in a [`BVec3A`].
#[inline(always)]
fn mask_to_bvec(m: <f32x4 as SimdLike>::Bool) -> BVec3A {
    BVec3A::from_mask(m)
}

/// # Basic Number Constants
impl Vec3A {
    /// All zeros.
    pub const ZERO: Self = Self::splat(0.0);

    /// All ones.
    pub const ONE: Self = Self::splat(1.0);

    /// All negative ones.
    pub const NEG_ONE: Self = Self::splat(-1.0);

    /// All `f32::MIN`.
    pub const MIN: Self = Self::splat(f32::MIN);

    /// All `f32::MAX`.
    pub const MAX: Self = Self::splat(f32::MAX);

    /// All `f32::NAN`.
    pub const NAN: Self = Self::splat(f32::NAN);

    /// All `f32::INFINITY`.
    pub const INFINITY: Self = Self::splat(f32::INFINITY);

    /// All `f32::NEG_INFINITY`.
    pub const NEG_INFINITY: Self = Self::splat(f32::NEG_INFINITY);

    /// A unit vector pointing along the positive X axis.
    pub const X: Self = Self::new(1.0, 0.0, 0.0);

    /// A unit vector pointing along the positive Y axis.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);

    /// A unit vector pointing along the positive Z axis.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    /// A unit vector pointing along the negative X axis.
    pub const NEG_X: Self = Self::new(-1.0, 0.0, 0.0);

    /// A unit vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self::new(0.0, -1.0, 0.0);

    /// A unit vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self::new(0.0, 0.0, -1.0);

    /// The unit axes.
    pub const AXES: [Self; 3] = [Self::X, Self::Y, Self::Z];
}

/// # Construction
impl Vec3A {
    /// Creates a new vector.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        // The fourth lane is unspecified
        Self(f32x4::from_array([x, y, z, z]))
    }

    /// Creates a new vector with all elements set to `v`.
    #[inline(always)]
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self(f32x4::from_array([v; 4]))
    }

    /// Returns a vector containing each element of `self` modified by a mapping function `f`.
    #[inline]
    #[must_use]
    pub fn map<F>(self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        Self::new(f(self.x), f(self.y), f(self.z))
    }

    /// Creates a vector from the elements in `if_true` and `if_false`, selecting which to use
    /// based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select(boolean: bool, if_true: Self, if_false: Self) -> Self {
        if boolean { if_true } else { if_false }
    }

    /// Creates a vector from the elements in `if_true` and `if_false`, selecting which to use
    /// based on the elements of `mask`.
    ///
    /// A true element in the mask uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select_mask(mask: BVec3A, if_true: Self, if_false: Self) -> Self {
        Self(Select::select(mask.to_mask(), if_true.0, if_false.0))
    }

    /// Creates a new vector from an array.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    /// Returns the vector as an array.
    #[inline]
    #[must_use]
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Creates a vector from the first 3 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 3 elements long.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= 3);
        Self::new(slice[0], slice[1], slice[2])
    }

    /// Writes the elements of `self` to the first 3 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 3 elements long.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        slice[..3].copy_from_slice(&self.to_array());
    }

    /// Creates a 4D vector from `self` and the given `w` value.
    #[inline]
    #[must_use]
    pub fn extend(self, w: f32) -> GVec4<f32> {
        GVec4::new(self.x, self.y, self.z, w)
    }

    /// Creates a 2D vector from the `x` and `y` elements of `self`, discarding `z`.
    ///
    /// Truncation may also be performed by using [`self.xy()`](crate::vector::Vec3Swizzles::xy()).
    #[inline]
    #[must_use]
    pub fn truncate(self) -> GVec2<f32> {
        GVec2::new(self.x, self.y)
    }

    /// Creates a 3D vector from `self` with the given value of `x`.
    #[inline]
    #[must_use]
    pub fn with_x(mut self, x: f32) -> Self {
        self.x = x;
        self
    }

    /// Creates a 3D vector from `self` with the given value of `y`.
    #[inline]
    #[must_use]
    pub fn with_y(mut self, y: f32) -> Self {
        self.y = y;
        self
    }

    /// Creates a 3D vector from `self` with the given value of `z`.
    #[inline]
    #[must_use]
    pub fn with_z(mut self, z: f32) -> Self {
        self.z = z;
        self
    }

    /// Projects a homogeneous coordinate to 3D space by performing perspective divide.
    #[inline]
    #[must_use]
    pub fn from_homogeneous(v: GVec4<f32>) -> Self {
        GVec3::from_homogeneous(v).into()
    }

    /// Creates a homogeneous coordinate from `self`, equivalent to `self.extend(1.0)`.
    #[inline]
    #[must_use]
    pub fn to_homogeneous(self) -> GVec4<f32> {
        self.extend(1.0)
    }
}

/// # Element-Wise Equality
impl Vec3A {
    /// Returns a vector mask containing the result of a `==` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words, this computes `[self.x == rhs.x, self.y == rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmpeq(self, rhs: Self) -> BVec3A {
        mask_to_bvec(NumEq::num_eq(self.0, rhs.0))
    }

    /// Returns a vector mask containing the result of a `!=` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x != rhs.x, self.y != rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmpne(self, rhs: Self) -> BVec3A {
        mask_to_bvec(NumEq::num_ne(self.0, rhs.0))
    }
}

/// # Element-Wise Ordering
impl Vec3A {
    /// Returns a vector mask containing the result of a `>=` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x >= rhs.x, self.y >= rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmpge(self, rhs: Self) -> BVec3A {
        mask_to_bvec(NumOrd::num_ge(self.0, rhs.0))
    }

    /// Returns a vector mask containing the result of a `>` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x > rhs.x, self.y > rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmpgt(self, rhs: Self) -> BVec3A {
        mask_to_bvec(NumOrd::num_gt(self.0, rhs.0))
    }

    /// Returns a vector mask containing the result of a `<=` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x <= rhs.x, self.y <= rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmple(self, rhs: Self) -> BVec3A {
        mask_to_bvec(NumOrd::num_le(self.0, rhs.0))
    }

    /// Returns a vector mask containing the result of a `<` comparison for each element of
    /// `self` and `rhs`.
    ///
    /// In other words this computes `[self.x < rhs.x, self.y < rhs.y, ..]` for all elements.
    #[inline]
    #[must_use]
    pub fn cmplt(self, rhs: Self) -> BVec3A {
        mask_to_bvec(NumOrd::num_lt(self.0, rhs.0))
    }

    /// Returns a vector containing the minimum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[min(x, rhs.x), min(self.y, rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self(NumOrd::min(self.0, rhs.0))
    }

    /// Returns a vector containing the maximum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[max(self.x, rhs.x), max(self.y, rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self(NumOrd::max(self.0, rhs.0))
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
    pub fn min_element(self) -> f32 {
        NumOrd::min(NumOrd::min(self.x, self.y), self.z)
    }

    /// Returns the horizontal maximum of `self`.
    ///
    /// In other words this computes `max(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn max_element(self) -> f32 {
        NumOrd::max(NumOrd::max(self.x, self.y), self.z)
    }

    /// Returns a vector containing the minimum values for each element of `self` and `rhs`,
    /// using [`NumOrd::min_fast`].
    ///
    /// # Floating-Point Types
    ///
    /// This is faster than [`min`](Self::min), but handles NaN and signed zero differently.
    /// Given a component of `self` and `rhs`, it _always_ returns `rhs` if `self` does not
    /// compare less than `rhs`, even if either value is NaN or if the two values compare equal
    /// (such as for the case of +0.0 and -0.0).
    ///
    /// See [`NumOrd::min_fast`] for more details.
    #[inline]
    #[must_use]
    pub fn min_fast(self, rhs: Self) -> Self {
        Self(NumOrd::min_fast(self.0, rhs.0))
    }

    /// Returns a vector containing the maximum values for each element of `self` and `rhs`,
    /// using [`NumOrd::max_fast`].
    ///
    /// # Floating-Point Types
    ///
    /// This is faster than [`max`](Self::max), but handles NaN and signed zero differently.
    /// Given a component of `self` and `rhs`, it _always_ returns `rhs` if `self` does not
    /// compare greater than `rhs`, even if either value is NaN or if the two values compare equal
    /// (such as for the case of +0.0 and -0.0).
    ///
    /// See [`NumOrd::max_fast`] for more details.
    #[inline]
    #[must_use]
    pub fn max_fast(self, rhs: Self) -> Self {
        Self(NumOrd::max_fast(self.0, rhs.0))
    }

    /// Element-wise clamping of values, using [`min_fast`](Self::min_fast) and
    /// [`max_fast`](Self::max_fast).
    ///
    /// # Floating-Point Types
    ///
    /// This is faster than [`clamp`](Self::clamp), but handles NaN and signed zero differently.
    /// Given a component of `self`, `min`, and `max`, it _always_ returns `min` if `self` does not
    /// compare greater than `min`, and otherwise it _always_ returns `max` if `self` does not
    /// compare less than `max`, even if any of the values are NaN or if the two values compare equal
    /// (such as for the case of +0.0 and -0.0).
    ///
    /// See [`NumOrd::min_fast`] and [`NumOrd::clamp_fast`] for more details.
    #[inline]
    #[must_use]
    pub fn clamp_fast(self, min: Self, max: Self) -> Self {
        self.max_fast(min).min_fast(max)
    }

    /// Returns the horizontal minimum of `self`, using [`NumOrd::min_fast`].
    ///
    /// # Floating-Point Types
    ///
    /// This is faster than [`min_element`](Self::min_element), but handles NaN and signed zero differently.
    /// Given a component of `self`, it _always_ returns the first element that does not compare greater
    /// than the others, even if any of the values are NaN or if the two values compare equal
    /// (such as for the case of +0.0 and -0.0).
    ///
    /// See [`NumOrd::min_fast`] for more details.
    #[inline]
    #[must_use]
    pub fn min_element_fast(self) -> f32 {
        let m = NumOrd::min_fast(self.0, Shuffle4::shuffle::<1, 0, 2, 3>(self.0));
        let m = NumOrd::min_fast(m, Shuffle4::shuffle::<2, 2, 2, 2>(m));
        SimdLike::extract(&m, 0)
    }

    /// Returns the horizontal maximum of `self`, using [`NumOrd::max_fast`].
    ///
    /// # Floating-Point Types
    ///
    /// This is faster than [`max_element`](Self::max_element), but handles NaN and signed zero differently.
    /// Given a component of `self`, it _always_ returns the first element that does not compare less
    /// than the others, even if any of the values are NaN or if the two values compare equal
    /// (such as for the case of +0.0 and -0.0).
    ///
    /// See [`NumOrd::max_fast`] for more details.
    #[inline]
    #[must_use]
    pub fn max_element_fast(self) -> f32 {
        let m = NumOrd::max_fast(self.0, Shuffle4::shuffle::<1, 0, 2, 3>(self.0));
        let m = NumOrd::max_fast(m, Shuffle4::shuffle::<2, 2, 2, 2>(m));
        SimdLike::extract(&m, 0)
    }
}

/// # Number Operations
impl Vec3A {
    /// Computes the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        let m = self.0 * rhs.0;
        (SimdLike::extract(&m, 0) + SimdLike::extract(&m, 1)) + SimdLike::extract(&m, 2)
    }

    /// Returns a vector where every element is the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self::splat(self.dot(rhs))
    }

    /// Computes the cross product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        let lhs_zxy = Shuffle4::shuffle::<2, 0, 1, 3>(self.0);
        let rhs_zxy = Shuffle4::shuffle::<2, 0, 1, 3>(rhs.0);
        Self(Shuffle4::shuffle::<2, 0, 1, 3>(
            lhs_zxy * rhs.0 - self.0 * rhs_zxy,
        ))
    }

    /// Returns the sum of all elements of `self`.
    ///
    /// In other words, this computes `self.x + self.y + ..`.
    #[inline]
    #[must_use]
    pub fn element_sum(self) -> f32 {
        self.x + self.y + self.z
    }

    /// Returns the product of all elements of `self`.
    ///
    /// In other words, this computes `self.x * self.y * ..`.
    #[inline]
    #[must_use]
    pub fn element_product(self) -> f32 {
        self.x * self.y * self.z
    }

    /// Computes the squared length of `self`.
    ///
    /// This is faster than [`length`](Self::length) as it avoids a square root operation.
    #[inline]
    #[must_use]
    #[doc(alias = "magnitude2")]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Compute the squared [Euclidean distance] between two points in space.
    ///
    /// [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
    #[inline]
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> f32 {
        (self - rhs).length_squared()
    }

    /// Returns a vector containing the absolute value of each element of `self`.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self(Signed::abs(self.0))
    }

    /// Returns a vector with elements representing the sign of `self`.
    ///
    /// - `1.0` if the element is positive
    /// - `-1.0` if the element is negative
    /// - `NAN` if the element is `NAN` (only for [`Float`](gnum::num::Float) types)
    #[inline]
    #[must_use]
    pub fn signum(self) -> Self {
        Self(Signed::signum(self.0))
    }
}

/// # Real Operations
impl Vec3A {
    /// Returns a vector with a length no less than `min` and no more than `max`.
    #[inline]
    #[must_use]
    pub fn clamp_length(self, min: f32, max: f32) -> Self {
        let length = self.length();
        let scale = NumOrd::min(NumOrd::max(min / length, 1.0), max / length);
        self * scale
    }

    /// Returns a vector with a length no less than `min`.
    #[inline]
    #[must_use]
    pub fn clamp_length_min(self, min: f32) -> Self {
        let length = self.length();
        let scale = NumOrd::max(min / length, 1.0);
        self * scale
    }

    /// Returns a vector with a length no more than `max`.
    #[inline]
    #[must_use]
    pub fn clamp_length_max(self, max: f32) -> Self {
        let length = self.length();
        let scale = NumOrd::min(max / length, 1.0);
        self * scale
    }

    /// Returns a vector with signs of `rhs` and the magnitudes of `self`.
    #[inline]
    #[must_use]
    pub fn copysign(self, rhs: Self) -> Self {
        Self(Real::copysign(self.0, rhs.0))
    }

    /// Computes the length of `self`.
    #[inline]
    #[must_use]
    #[doc(alias = "magnitude")]
    pub fn length(self) -> f32 {
        Real::sqrt(self.length_squared())
    }

    /// Computes `1.0 / length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn length_recip(self) -> f32 {
        1.0 / self.length()
    }

    /// Computes the [Euclidean distance] between two points in space.
    ///
    /// [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
    #[inline]
    #[must_use]
    pub fn distance(self, rhs: Self) -> f32 {
        (self - rhs).length()
    }

    /// Returns `self` normalized to length `1.0`.
    ///
    /// For valid results, `self` must be finite and _not_ of length zero, nor very close to zero.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        self * self.length_recip()
    }

    /// Returns whether `self` is length `1.0` or not, within a certain `eps` tolerance.
    #[inline]
    #[must_use]
    pub fn is_normalized(self, eps: f32) -> bool {
        Signed::abs(self.length_squared() - 1.0) <= eps
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

    /// Returns a vector containing the integer nearest to a number for each element of `self`.
    /// If a value is half-way between two integers, rounds away from zero.
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        Self(Real::round(self.0))
    }

    /// Returns a vector containing the integer nearest to a number for each element of `self`.
    /// If a value is half-way between two integers, rounds to the number with an even least significant digit.
    #[inline]
    #[must_use]
    pub fn round_ties_even(self) -> Self {
        Self(Real::round_ties_even(self.0))
    }

    /// Returns a vector containing the largest integer less than or equal to a number for each
    /// element of `self`.
    #[inline]
    #[must_use]
    pub fn floor(self) -> Self {
        Self(Real::floor(self.0))
    }

    /// Returns a vector containing the smallest integer greater than or equal to a number for
    /// each element of `self`.
    #[inline]
    #[must_use]
    pub fn ceil(self) -> Self {
        Self(Real::ceil(self.0))
    }

    /// Returns a vector containing the integer part each element of `self`. This means numbers are
    /// always truncated towards zero.
    #[inline]
    #[must_use]
    pub fn trunc(self) -> Self {
        Self(Real::trunc(self.0))
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

    /// Returns a vector containing `e^self` (the exponential function) for each element of `self`.
    ///
    /// # Unspecified Precision
    ///
    /// The precision of this function is non-deterministic. This means it varies
    /// by platform, Rust version, and can even differ within the same execution
    /// from one invocation to the next.
    ///
    /// See [`exp_stable`](Self::exp_stable) for a version of this function
    /// that is guaranteed to be deterministic and returns identical results
    /// across both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    #[inline]
    #[must_use]
    pub fn exp(self) -> Self {
        Self::new(Real::exp(self.x), Real::exp(self.y), Real::exp(self.z))
    }

    /// Returns a vector containing `e^self` (the exponential function) for each element of `self`,
    /// with deterministic results.
    ///
    /// # Precision
    ///
    /// This function is deterministic and returns identical results across
    /// both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    ///
    /// See [`exp`](Self::exp) for a version of this function that may be
    /// more precise but can be non-deterministic.
    #[inline]
    #[must_use]
    pub fn exp_stable(self) -> Self {
        Self(Real::exp_stable(self.0))
    }

    /// Returns a vector containing `2^self` for each element of `self`.
    ///
    /// # Unspecified Precision
    ///
    /// The precision of this function is non-deterministic. This means it varies
    /// by platform, Rust version, and can even differ within the same execution
    /// from one invocation to the next.
    ///
    /// See [`exp2_stable`](Self::exp2_stable) for a version of this function
    /// that is guaranteed to be deterministic and returns identical results
    /// across both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    #[inline]
    #[must_use]
    pub fn exp2(self) -> Self {
        Self::new(Real::exp2(self.x), Real::exp2(self.y), Real::exp2(self.z))
    }

    /// Returns a vector containing `2^self` for each element of `self`,
    /// with deterministic results.
    ///
    /// # Precision
    ///
    /// This function is deterministic and returns identical results across
    /// both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    ///
    /// See [`exp2`](Self::exp2) for a version of this function that may be
    /// more precise but can be non-deterministic.
    #[inline]
    #[must_use]
    pub fn exp2_stable(self) -> Self {
        Self(Real::exp2_stable(self.0))
    }

    /// Returns a vector containing the natural logarithm for each element of `self`.
    /// This returns NaN when the element is negative and negative infinity
    /// when the element is zero.
    ///
    /// # Unspecified Precision
    ///
    /// The precision of this function is non-deterministic. This means it varies
    /// by platform, Rust version, and can even differ within the same execution
    /// from one invocation to the next.
    ///
    /// See [`ln_stable`](Self::ln_stable) for a version of this function
    /// that is guaranteed to be deterministic and returns identical results
    /// across both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    #[inline]
    #[must_use]
    pub fn ln(self) -> Self {
        Self(Real::ln(self.0))
    }

    /// Returns a vector containing the natural logarithm for each element of `self`,
    /// with deterministic results. This returns NaN when the element is negative
    /// and negative infinity when the element is zero.
    ///
    /// # Precision
    ///
    /// This function is deterministic and returns identical results across
    /// both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    ///
    /// See [`ln`](Self::ln) for a version of this function that may be
    /// more precise but can be non-deterministic.
    #[inline]
    #[must_use]
    pub fn ln_stable(self) -> Self {
        Self(Real::ln_stable(self.0))
    }

    /// Returns a vector containing the base 2 logarithm for each element of `self`.
    /// This returns NaN when the element is negative and negative infinity
    /// when the element is zero.
    ///
    /// # Unspecified Precision
    ///
    /// The precision of this function is non-deterministic. This means it varies
    /// by platform, Rust version, and can even differ within the same execution
    /// from one invocation to the next.
    ///
    /// See [`log2_stable`](Self::log2_stable) for a version of this function
    /// that is guaranteed to be deterministic and returns identical results
    /// across both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    #[inline]
    #[must_use]
    pub fn log2(self) -> Self {
        Self(Real::log2(self.0))
    }

    /// Returns a vector containing the base 2 logarithm for each element of `self`,
    /// with deterministic results. This returns NaN when the element is negative
    /// and negative infinity when the element is zero.
    ///
    /// # Precision
    ///
    /// This function is deterministic and returns identical results across
    /// both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    ///
    /// See [`log2`](Self::log2) for a version of this function that may be
    /// more precise but can be non-deterministic.
    #[inline]
    #[must_use]
    pub fn log2_stable(self) -> Self {
        Self(Real::log2_stable(self.0))
    }

    /// Returns a vector containing the square root for each element of `self`.
    /// This returns NaN when the element is negative.
    #[inline]
    #[must_use]
    pub fn sqrt(self) -> Self {
        Self(Real::sqrt(self.0))
    }

    /// Returns a vector containing the cosine for each element of `self`.
    ///
    /// # Unspecified Precision
    ///
    /// The precision of this function is non-deterministic. This means it varies
    /// by platform, Rust version, and can even differ within the same execution
    /// from one invocation to the next.
    ///
    /// See [`cos_stable`](Self::cos_stable) for a version of this function
    /// that is guaranteed to be deterministic and returns identical results
    /// across both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    #[inline]
    #[must_use]
    pub fn cos(self) -> Self {
        Self(Real::cos(self.0))
    }

    /// Returns a vector containing the cosine for each element of `self`,
    /// with deterministic results.
    ///
    /// # Precision
    ///
    /// This function is deterministic and returns identical results across
    /// both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    ///
    /// See [`cos`](Self::cos) for a version of this function that may be
    /// more precise but can be non-deterministic.
    #[inline]
    #[must_use]
    pub fn cos_stable(self) -> Self {
        Self(Real::cos_stable(self.0))
    }

    /// Returns a vector containing the sine for each element of `self`.
    ///
    /// # Unspecified Precision
    ///
    /// The precision of this function is non-deterministic. This means it varies
    /// by platform, Rust version, and can even differ within the same execution
    /// from one invocation to the next.
    ///
    /// See [`sin_stable`](Self::sin_stable) for a version of this function
    /// that is guaranteed to be deterministic and returns identical results
    /// across both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    #[inline]
    #[must_use]
    pub fn sin(self) -> Self {
        Self(Real::sin(self.0))
    }

    /// Returns a vector containing the sine for each element of `self`,
    /// with deterministic results.
    ///
    /// # Precision
    ///
    /// This function is deterministic and returns identical results across
    /// both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    ///
    /// See [`sin`](Self::sin) for a version of this function that may be
    /// more precise but can be non-deterministic.
    #[inline]
    #[must_use]
    pub fn sin_stable(self) -> Self {
        Self(Real::sin_stable(self.0))
    }

    /// Returns a tuple of two vectors containing the sine and cosine for each element of `self`.
    ///
    /// # Unspecified Precision
    ///
    /// The precision of this function is non-deterministic. This means it varies
    /// by platform, Rust version, and can even differ within the same execution
    /// from one invocation to the next.
    ///
    /// See [`sin_cos_stable`](Self::sin_cos_stable) for a version of this function
    /// that is guaranteed to be deterministic and returns identical results
    /// across both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    #[inline]
    #[must_use]
    pub fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = Real::sin_cos(self.0);
        (Self(sin), Self(cos))
    }

    /// Returns a tuple of two vectors containing the sine and cosine for each element of `self`,
    /// with deterministic results.
    ///
    /// # Precision
    ///
    /// This function is deterministic and returns identical results across
    /// both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    ///
    /// See [`sin_cos`](Self::sin_cos) for a version of this function that may be
    /// more precise but can be non-deterministic.
    #[inline]
    #[must_use]
    pub fn sin_cos_stable(self) -> (Self, Self) {
        let (sin, cos) = Real::sin_cos_stable(self.0);
        (Self(sin), Self(cos))
    }

    /// Returns a vector containing the reciprocal `1.0 / n` of each element of `self`.
    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        Self(Real::recip(self.0))
    }

    /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
    /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
    /// extrapolated.
    #[inline]
    #[must_use]
    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        self * (1.0 - s) + rhs * s
    }

    /// Calculates the midpoint between `self` and `rhs`.
    ///
    /// The midpoint is the average of, or halfway point between, two vectors.
    /// `a.midpoint(b)` should yield the same result as `a.lerp(b, 0.5)`
    /// while being slightly cheaper to compute.
    #[inline]
    #[must_use]
    pub fn midpoint(self, rhs: Self) -> Self {
        Self(Real::midpoint(self.0, rhs.0))
    }

    /// Calculates the midpoint between `self` and `rhs` as `(self + rhs) / 2`.
    ///
    /// # Floating-Point Types
    ///
    /// This is a faster version of [`midpoint`](Self::midpoint), which additionally guards
    /// against overflow and underflow. For example, `Vec3A::MAX.midpoint(Vec3A::MAX)` is `Vec3A::MAX`,
    /// but `Vec3A::MAX.midpoint_fast(Vec3A::MAX)` overflows to `Vec3A::INFINITY`.
    #[inline]
    #[must_use]
    pub fn midpoint_fast(self, rhs: Self) -> Self {
        Self(Real::midpoint_fast(self.0, rhs.0))
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
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        let d = (self - rhs).abs();
        d.x <= max_abs_diff && d.y <= max_abs_diff && d.z <= max_abs_diff
    }

    /// Returns the reflection vector for a given incident vector `self` and surface normal
    /// `normal`.
    ///
    /// `normal` must be normalized.
    #[inline]
    #[must_use]
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * (2.0 * self.dot(normal))
    }

    /// Returns the refraction direction for a given incident vector `self`, surface normal
    /// `normal` and ratio of indices of refraction, `eta`. When total internal reflection occurs,
    /// a zero vector will be returned.
    ///
    /// `self` and `normal` must be normalized.
    #[inline]
    #[must_use]
    pub fn refract(self, normal: Self, eta: f32) -> Self {
        let ndi = normal.dot(self);
        let k = 1.0 - eta * eta * (1.0 - ndi * ndi);
        // Total internal reflection is `k < 0`, where the square root would be imaginary.
        if k > 0.0 {
            self * eta - normal * (eta * ndi + Real::sqrt(k))
        } else {
            Self::ZERO
        }
    }

    /// Returns the angle (in radians) between two vectors in the range `[0, +π]`.
    ///
    /// The inputs do not need to be unit vectors however they must be non-zero.
    #[inline]
    #[must_use]
    pub fn angle_between(self, rhs: Self) -> f32 {
        let dot = self.dot(rhs) / Real::sqrt(self.length_squared() * rhs.length_squared());
        Real::acos_stable(NumOrd::clamp(dot, -1.0, 1.0))
    }

    /// Rotates around the x axis by `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn rotate_x(self, angle: f32) -> Self {
        let (sina, cosa) = Real::sin_cos_stable(angle);
        Self::new(
            self.x,
            self.y * cosa - self.z * sina,
            self.y * sina + self.z * cosa,
        )
    }

    /// Rotates around the y axis by `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn rotate_y(self, angle: f32) -> Self {
        let (sina, cosa) = Real::sin_cos_stable(angle);
        Self::new(
            self.x * cosa + self.z * sina,
            self.y,
            self.x * -sina + self.z * cosa,
        )
    }

    /// Rotates around the z axis by `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn rotate_z(self, angle: f32) -> Self {
        let (sina, cosa) = Real::sin_cos_stable(angle);
        Self::new(
            self.x * cosa - self.y * sina,
            self.x * sina + self.y * cosa,
            self.z,
        )
    }

    /// Rotates around `axis` by `angle` (in radians).
    ///
    /// The axis must be a unit vector.
    #[inline]
    #[must_use]
    pub fn rotate_axis(self, axis: Self, angle: f32) -> Self {
        Rot3A::from_axis_angle_vec3a(axis, angle).mul_vec3a(self)
    }
}

/// # Float Methods
impl Vec3A {
    /// Returns true if, and only if, all elements are finite.  If any element is either
    /// `NaN`, positive or negative infinity, this will return false.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        Float::is_finite(self.x) && Float::is_finite(self.y) && Float::is_finite(self.z)
    }

    /// Performs [`is_finite`](gnum::num::Float::is_finite) on each element of self, returning a vector mask of the results.
    ///
    /// In other words, this computes `[x.is_finite(), y.is_finite(), ...]`.
    #[inline]
    #[must_use]
    pub fn is_finite_mask(self) -> BVec3A {
        mask_to_bvec(NumOrd::num_lt(self.0.abs(), Self::INFINITY.0))
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        Float::is_nan(self.x) || Float::is_nan(self.y) || Float::is_nan(self.z)
    }

    /// Performs [`is_nan`](gnum::num::Float::is_nan) on each element of self, returning a vector mask of the results.
    ///
    /// In other words, this computes `[x.is_nan(), y.is_nan(), ...]`.
    #[inline]
    #[must_use]
    pub fn is_nan_mask(self) -> BVec3A {
        mask_to_bvec(NumEq::num_ne(self.0, self.0))
    }

    /// Returns `self` normalized to length `1.0` if possible, else returns a fallback value.
    ///
    /// In particular, if the input is zero (or very close to zero), or non-finite,
    /// the result of this operation will be the fallback value.
    #[inline]
    #[must_use]
    pub fn normalize_or(self, fallback: Self) -> Self {
        let rcp = self.length_recip();
        if Float::is_finite(rcp) && rcp > 0.0 {
            self * rcp
        } else {
            fallback
        }
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
    /// If `self` is zero length, then `(Self::X, 0.0)` is returned.
    #[inline]
    #[must_use]
    pub fn normalize_and_length(self) -> (Self, f32) {
        let length = self.length();
        let rcp = 1.0 / length;
        if Float::is_finite(rcp) && rcp > 0.0 {
            (self * rcp, length)
        } else {
            (Self::X, 0.0)
        }
    }

    /// Returns a vector containing each element of `self` raised to the power of `n`.
    ///
    /// # Unspecified Precision
    ///
    /// The precision of this function is non-deterministic. This means it varies
    /// by platform, Rust version, and can even differ within the same execution
    /// from one invocation to the next.
    ///
    /// See [`powf_stable`](Self::powf_stable) for a version of this function
    /// that is guaranteed to be deterministic and returns identical results
    /// across both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    #[inline]
    #[must_use]
    pub fn powf(self, n: f32) -> Self {
        Self::new(
            Float::powf(self.x, n),
            Float::powf(self.y, n),
            Float::powf(self.z, n),
        )
    }

    /// Returns a vector containing each element of `self` raised to the power of `n`,
    /// with deterministic results.
    ///
    /// # Precision
    ///
    /// This function is deterministic and returns identical results across
    /// both scalar and vectorized types at the cost of some precision
    /// and/or performance.
    ///
    /// See [`powf`](Self::powf) for a version of this function that may be
    /// more precise but can be non-deterministic.
    #[inline]
    #[must_use]
    pub fn powf_stable(self, n: f32) -> Self {
        Self(Float::powf_stable(self.0, f32x4::splat(n)))
    }

    /// Moves towards `rhs` based on the value `d`.
    ///
    /// When `d` is `0.0`, the result will be equal to `self`. When `d` is equal to
    /// `self.distance(rhs)`, the result will be equal to `rhs`. Will not go past `rhs`.
    #[inline]
    #[must_use]
    pub fn move_towards(self, rhs: Self, d: f32) -> Self {
        let a = rhs - self;
        let len = a.length();
        if len <= d || len <= f32::EPSILON {
            rhs
        } else {
            self + a / len * d
        }
    }

    /// Rotates towards `rhs` up to `max_angle` (in radians).
    ///
    /// When `max_angle` is `0.0`, the result will be equal to `self`. When `max_angle` is equal to
    /// `self.angle_between(rhs)`, the result will be parallel to `rhs`. If `max_angle` is negative,
    /// rotates towards the exact opposite of `rhs`. Will not go past the target.
    #[inline]
    #[must_use]
    pub fn rotate_towards(self, rhs: Self, max_angle: f32) -> Self {
        let angle_between = self.angle_between(rhs);
        // When `max_angle < 0`, rotate no further than `PI` radians away
        let angle = NumOrd::clamp(
            max_angle,
            angle_between - core::f32::consts::PI,
            angle_between,
        );
        let axis = self
            .cross(rhs)
            .normalize_or(self.any_orthogonal_vector().normalize());
        Rot3A::from_axis_angle_vec3a(axis, angle).mul_vec3a(self)
    }

    /// Returns some vector that is orthogonal to the given one.
    ///
    /// The input vector must be finite and non-zero.
    ///
    /// The output vector is not necessarily unit length. For that use
    /// [`Self::any_orthonormal_vector()`] instead.
    #[inline]
    #[must_use]
    pub fn any_orthogonal_vector(self) -> Self {
        if Signed::abs(self.x) > Signed::abs(self.y) {
            Self::new(-self.z, 0.0, self.x)
        } else {
            Self::new(0.0, self.z, -self.y)
        }
    }

    /// Returns any unit vector that is orthogonal to the given one.
    ///
    /// The input vector must be unit length.
    #[inline]
    #[must_use]
    pub fn any_orthonormal_vector(self) -> Self {
        // From https://graphics.pixar.com/library/OrthonormalB/paper.pdf
        let sign = Signed::signum(self.z);
        let a = -1.0 / (sign + self.z);
        let b = self.x * self.y * a;
        Self::new(b, sign + self.y * self.y * a, -self.y)
    }

    /// Given a unit vector return two other vectors that together form an orthonormal
    /// basis. That is, all three vectors are orthogonal to each other and are normalized.
    #[inline]
    #[must_use]
    pub fn any_orthonormal_pair(self) -> (Self, Self) {
        // From https://graphics.pixar.com/library/OrthonormalB/paper.pdf
        let sign = Signed::signum(self.z);
        let a = -1.0 / (sign + self.z);
        let b = self.x * self.y * a;
        (
            Self::new(1.0 + sign * self.x * self.x * a, sign * b, -sign * self.x),
            Self::new(b, sign + self.y * self.y * a, -self.y),
        )
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
        Self(Float::mul_add(self.0, a.0, b.0))
    }

    /// Performs a spherical linear interpolation between `self` and `rhs` based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
    /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
    /// extrapolated.
    #[inline]
    #[must_use]
    pub fn slerp(self, rhs: Self, s: f32) -> Self {
        let self_length = self.length();
        let rhs_length = rhs.length();
        // Cosine of the angle between the vectors [-1, 1], or NaN if either vector has a zero length
        let dot = self.dot(rhs) / (self_length * rhs_length);
        // If dot is close to 1 or -1, or is NaN the calculations for t1 and t2 break down
        if Signed::abs(dot) < 1.0 - f32::EPSILON {
            // Angle between the vectors [0, +π]
            let theta = Real::acos_stable(NumOrd::clamp(dot, -1.0, 1.0));
            // Sine of the angle between vectors [0, 1]
            let sins = Real::sin_stable(f32x4::from_array([
                theta,
                theta * (1.0 - s),
                theta * s,
                theta,
            ]));
            let sin_theta = SimdLike::extract(&sins, 0);
            let t1 = SimdLike::extract(&sins, 1);
            let t2 = SimdLike::extract(&sins, 2);

            // Interpolate vector lengths
            let result_length = self_length + (rhs_length - self_length) * s;
            // Scale the vectors to the target length and interpolate them
            return (self * (result_length / self_length) * t1
                + rhs * (result_length / rhs_length) * t2)
                * Real::recip(sin_theta);
        }
        if dot < 0.0 {
            // Vectors are almost parallel in opposing directions

            // Create a rotation from self to rhs along some axis
            let axis = self.any_orthogonal_vector().normalize();
            let rotation = Rot3A::from_axis_angle_vec3a(axis, core::f32::consts::PI * s);
            // Interpolate vector lengths
            let result_length = self_length + (rhs_length - self_length) * s;
            rotation.mul_vec3a(self) * (result_length / self_length)
        } else {
            // Vectors are almost parallel in the same direction, or dot was NaN
            self.lerp(rhs, s)
        }
    }
}

/// # Conversions
impl Vec3A {
    /// Creates a [`Vec3A`] from a [`GVec3<f32>`].
    #[inline(always)]
    #[must_use]
    pub const fn from_vec3(v: GVec3<f32>) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Converts `self` to a [`GVec3<f32>`].
    #[inline(always)]
    #[must_use]
    pub fn to_vec3(self) -> GVec3<f32> {
        GVec3::new(self.x, self.y, self.z)
    }

    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Copy>(self) -> GVec3<U>
    where
        f32: NumCast<U>,
    {
        self.to_vec3().cast()
    }
}

impl Vec3A {
    /// Creates a new [`Vec3A`] from its SIMD register.
    #[inline(always)]
    pub(crate) const fn from_register(r: f32x4) -> Self {
        Self(r)
    }
}

impl Deref for Vec3A {
    type Target = crate::deref::Vec3<f32>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl DerefMut for Vec3A {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl PartialEq for Vec3A {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
    }
}

impl Default for Vec3A {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

macro_rules! impl_binary_op {
    ($($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident);* $(;)?) => {
        $(
            impl $trait for Vec3A {
                type Output = Self;

                #[inline(always)]
                fn $method(self, rhs: Self) -> Self {
                    Self($trait::$method(self.0, rhs.0))
                }
            }

            impl $trait<&Vec3A> for Vec3A {
                type Output = Self;

                #[inline(always)]
                fn $method(self, rhs: &Vec3A) -> Self {
                    $trait::$method(self, *rhs)
                }
            }

            impl $trait<Vec3A> for &Vec3A {
                type Output = Vec3A;

                #[inline(always)]
                fn $method(self, rhs: Vec3A) -> Vec3A {
                    $trait::$method(*self, rhs)
                }
            }

            impl $trait<&Vec3A> for &Vec3A {
                type Output = Vec3A;

                #[inline(always)]
                fn $method(self, rhs: &Vec3A) -> Vec3A {
                    $trait::$method(*self, *rhs)
                }
            }

            impl $trait<f32> for Vec3A {
                type Output = Self;

                #[inline(always)]
                fn $method(self, rhs: f32) -> Self {
                    Self($trait::$method(self.0, f32x4::splat(rhs)))
                }
            }

            impl $trait<&f32> for Vec3A {
                type Output = Self;

                #[inline(always)]
                fn $method(self, rhs: &f32) -> Self {
                    $trait::$method(self, *rhs)
                }
            }

            impl $trait<f32> for &Vec3A {
                type Output = Vec3A;

                #[inline(always)]
                fn $method(self, rhs: f32) -> Vec3A {
                    $trait::$method(*self, rhs)
                }
            }

            impl $trait<&f32> for &Vec3A {
                type Output = Vec3A;

                #[inline(always)]
                fn $method(self, rhs: &f32) -> Vec3A {
                    $trait::$method(*self, *rhs)
                }
            }

            impl $trait<Vec3A> for f32 {
                type Output = Vec3A;

                #[inline(always)]
                fn $method(self, rhs: Vec3A) -> Vec3A {
                    Vec3A($trait::$method(f32x4::splat(self), rhs.0))
                }
            }

            impl $trait<&Vec3A> for f32 {
                type Output = Vec3A;

                #[inline(always)]
                fn $method(self, rhs: &Vec3A) -> Vec3A {
                    $trait::$method(self, *rhs)
                }
            }

            impl $trait<Vec3A> for &f32 {
                type Output = Vec3A;

                #[inline(always)]
                fn $method(self, rhs: Vec3A) -> Vec3A {
                    $trait::$method(*self, rhs)
                }
            }

            impl $trait<&Vec3A> for &f32 {
                type Output = Vec3A;

                #[inline(always)]
                fn $method(self, rhs: &Vec3A) -> Vec3A {
                    $trait::$method(*self, *rhs)
                }
            }

            impl $assign_trait for Vec3A {
                #[inline(always)]
                fn $assign_method(&mut self, rhs: Self) {
                    *self = $trait::$method(*self, rhs);
                }
            }

            impl $assign_trait<&Vec3A> for Vec3A {
                #[inline(always)]
                fn $assign_method(&mut self, rhs: &Vec3A) {
                    $assign_trait::$assign_method(self, *rhs);
                }
            }

            impl $assign_trait<f32> for Vec3A {
                #[inline(always)]
                fn $assign_method(&mut self, rhs: f32) {
                    *self = $trait::$method(*self, rhs);
                }
            }

            impl $assign_trait<&f32> for Vec3A {
                #[inline(always)]
                fn $assign_method(&mut self, rhs: &f32) {
                    $assign_trait::$assign_method(self, *rhs);
                }
            }
        )*
    };
}

impl_binary_op!(
    Add, add, AddAssign, add_assign;
    Sub, sub, SubAssign, sub_assign;
    Mul, mul, MulAssign, mul_assign;
    Div, div, DivAssign, div_assign;
    Rem, rem, RemAssign, rem_assign;
);

macro_rules! impl_euclid_op {
    ($($trait:ident, $method:ident);* $(;)?) => {
        $(
            impl $trait for Vec3A {
                type Output = Self;

                #[inline]
                fn $method(self, rhs: Self) -> Self {
                    Self($trait::$method(self.0, rhs.0))
                }
            }

            impl $trait<&Vec3A> for Vec3A {
                type Output = Self;

                #[inline]
                fn $method(self, rhs: &Vec3A) -> Self {
                    $trait::$method(self, *rhs)
                }
            }

            impl $trait<Vec3A> for &Vec3A {
                type Output = Vec3A;

                #[inline]
                fn $method(self, rhs: Vec3A) -> Vec3A {
                    $trait::$method(*self, rhs)
                }
            }

            impl $trait<&Vec3A> for &Vec3A {
                type Output = Vec3A;

                #[inline]
                fn $method(self, rhs: &Vec3A) -> Vec3A {
                    $trait::$method(*self, *rhs)
                }
            }
        )*
    };
}

impl_euclid_op!(
    DivEuclid, div_euclid;
    RemEuclid, rem_euclid;
);

impl Neg for Vec3A {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Neg for &Vec3A {
    type Output = Vec3A;

    #[inline(always)]
    fn neg(self) -> Vec3A {
        -*self
    }
}

impl Sum<Vec3A> for Vec3A {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a> Sum<&'a Vec3A> for Vec3A {
    #[inline]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl Product<Vec3A> for Vec3A {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl<'a> Product<&'a Vec3A> for Vec3A {
    #[inline]
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, &b| a * b)
    }
}

impl Index<usize> for Vec3A {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vec3A {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds"),
        }
    }
}

impl From<GVec3<f32>> for Vec3A {
    #[inline(always)]
    fn from(v: GVec3<f32>) -> Self {
        Self::from_vec3(v)
    }
}

impl From<Vec3A> for GVec3<f32> {
    #[inline(always)]
    fn from(v: Vec3A) -> Self {
        v.to_vec3()
    }
}

impl From<[f32; 3]> for Vec3A {
    #[inline(always)]
    fn from(arr: [f32; 3]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec3A> for [f32; 3] {
    #[inline(always)]
    fn from(v: Vec3A) -> Self {
        v.to_array()
    }
}

impl From<(f32, f32, f32)> for Vec3A {
    #[inline(always)]
    fn from((x, y, z): (f32, f32, f32)) -> Self {
        Self::new(x, y, z)
    }
}

impl From<Vec3A> for (f32, f32, f32) {
    #[inline(always)]
    fn from(v: Vec3A) -> Self {
        (v.x, v.y, v.z)
    }
}

impl From<(GVec2<f32>, f32)> for Vec3A {
    #[inline(always)]
    fn from((v, z): (GVec2<f32>, f32)) -> Self {
        Self::new(v.x, v.y, z)
    }
}

impl AsRef<[f32; 3]> for Vec3A {
    #[inline]
    fn as_ref(&self) -> &[f32; 3] {
        unsafe { &*(self as *const Vec3A as *const [f32; 3]) }
    }
}

impl AsMut<[f32; 3]> for Vec3A {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32; 3] {
        unsafe { &mut *(self as *mut Vec3A as *mut [f32; 3]) }
    }
}

impl fmt::Display for Vec3A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.x, self.y, self.z)
    }
}

impl fmt::Debug for Vec3A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec3A")
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .finish()
    }
}
