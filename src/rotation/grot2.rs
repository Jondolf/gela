use crate::affine::GAffine2;
use crate::matrix::GMat2;
use crate::vector::GVec2;

use core::{
    iter::{Product, Sum},
    ops::*,
};

use gnum::{
    num::{Float, Real},
    simd::Select,
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// Creates a 2D rotation from the cosine and sine of the angle (in radians).
///
/// This should generally not be called manually unless you know what you are doing.
/// Use one of the other constructors instead such as [`from_radians`] or [`from_degrees`].
///
/// [`from_radians`]: GRot2::from_radians
/// [`from_degrees`]: GRot2::from_degrees
#[inline(always)]
#[must_use]
pub const fn grot2<T: Real>(cos: T, sin: T) -> GRot2<T> {
    GRot2::from_cos_sin(cos, sin)
}

/// A 2D rotation represented as a unit complex number.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[cfg_attr(
    feature = "zerocopy",
    derive(FromBytes, Immutable, IntoBytes, KnownLayout)
)]
#[cfg_attr(feature = "cuda", repr(align(8)))]
#[repr(C)]
#[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
pub struct GRot2<T: Real> {
    /// The cosine of the rotation angle (in radians).
    ///
    /// This is the real part of the unit complex number.
    pub cos: T,
    /// The sine of the rotation angle (in radians).
    ///
    /// This is the imaginary part of the unit complex number.
    pub sin: T,
}

/// # Basic Number Constants
impl<T: Real> GRot2<T> {
    /// All zeros.
    pub const ZERO: Self = Self::from_cos_sin(T::ZERO, T::ZERO);

    /// The identity rotation. Corresponds to no rotation.
    pub const IDENTITY: Self = Self::from_cos_sin(T::ONE, T::ZERO);

    /// A rotation of π radians (180 degrees).
    pub const PI: Self = Self::from_cos_sin(T::NEG_ONE, T::ZERO);

    /// A rotation of π/2 radians (90 degrees).
    pub const FRAC_PI_2: Self = Self::from_cos_sin(T::ZERO, T::ONE);

    /// A rotation of π/4 radians (45 degrees).
    pub const FRAC_PI_4: Self = Self::from_cos_sin(T::FRAC_1_SQRT_2, T::FRAC_1_SQRT_2);
}

/// # Float Constants
impl<T: Float> GRot2<T> {
    /// All `NaN`.
    pub const NAN: Self = Self::from_cos_sin(T::NAN, T::NAN);
}

/// # Constructors
impl<T: Real> GRot2<T> {
    /// Creates a new 2D rotation.
    ///
    /// This should generally not be called manually unless you know what you are doing.
    /// Use one of the other constructors instead such as `from_radians`.
    ///
    /// `from_cos_sin` is mostly used by unit tests and `serde` deserialization.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    #[inline(always)]
    #[must_use]
    pub const fn from_cos_sin(cos: T, sin: T) -> Self {
        Self { cos, sin }
    }

    /// Creates a 2D rotation from the elements in `if_true` and `if_false`, selecting which to use
    /// based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select<B: Select<T>>(boolean: B, if_true: Self, if_false: Self) -> Self {
        Self {
            cos: boolean.select(if_true.cos, if_false.cos),
            sin: boolean.select(if_true.sin, if_false.sin),
        }
    }

    /// Creates a new 2D rotation from an array in the form `[cos, sin]`.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    #[inline]
    #[must_use]
    pub const fn from_array(a: [T; 2]) -> Self {
        Self::from_cos_sin(a[0], a[1])
    }

    /// Creates a new 2D rotation from a 2D vector in the form `[cos, sin]`.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    #[inline]
    #[must_use]
    pub const fn from_vec2(v: GVec2<T>) -> Self {
        Self { cos: v.x, sin: v.y }
    }

    /// Creates a 2D rotation from a slice in the form `[cos, sin]`.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    ///
    /// # Panics
    ///
    /// Panics if `slice` length is less than 2.
    #[inline]
    #[must_use]
    pub fn from_slice(slice: &[T]) -> Self {
        Self::from_cos_sin(slice[0], slice[1])
    }

    /// Writes the rotation to an unaligned slice.
    ///
    /// # Panics
    ///
    /// Panics if `slice` length is less than 2.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [T]) {
        slice[0] = self.cos;
        slice[1] = self.sin;
    }

    /// Creates a 2D rotation from an angle in radians.
    #[inline]
    #[must_use]
    pub fn from_radians(angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::from_cos_sin(cos, sin)
    }

    /// Creates a 2D rotation from an angle in degrees.
    #[inline]
    #[must_use]
    pub fn from_degrees(angle: T) -> Self {
        let (sin, cos) = angle.to_radians().sin_cos();
        Self::from_cos_sin(cos, sin)
    }

    /// Creates a 2D rotation from a 2x2 rotation matrix.
    ///
    /// Note if the input matrix contain scales, shears, or other non-rotation transformations,
    /// the resulting 2D rotation will be ill-defined.
    #[inline]
    #[must_use]
    pub fn from_mat2(mat: &GMat2<T>) -> Self {
        Self::from_cos_sin(mat.x_axis.x, mat.y_axis.x)
    }

    /// Returns the rotation angle of `self` in radians.
    #[inline]
    #[must_use]
    pub fn to_radians(self) -> T {
        self.sin.atan2(self.cos)
    }

    /// Returns the rotation angle of `self` in degrees.
    #[inline]
    #[must_use]
    pub fn to_degrees(self) -> T {
        self.to_radians().to_degrees()
    }

    /// `[cos, sin]`
    #[inline]
    #[must_use]
    pub fn to_array(self) -> [T; 2] {
        [self.cos, self.sin]
    }

    /// Returns the conjugate of `self`. For a unit complex number the
    /// conjugate is also the inverse.
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self::from_cos_sin(self.cos, -self.sin)
    }

    /// Returns the inverse of a normalized 2D rotation.
    ///
    /// Typically, the inverse returns the conjugate of a normalized 2D rotation.
    /// Because `self` is assumed to already be unit length, this method *does not*
    /// normalize before returning the conjugate.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        self.conjugate()
    }

    /// Computes the dot product of `self` and `rhs`. The dot product is
    /// equal to the cosine of the angle between two 2D rotations.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> T {
        GVec2::from(self).dot(GVec2::from(rhs))
    }

    /// Computes the length of `self`.
    #[doc(alias = "magnitude")]
    #[inline]
    #[must_use]
    pub fn length(self) -> T {
        GVec2::from(self).length()
    }

    /// Computes the squared length of `self`.
    ///
    /// This is generally faster than `length()` as it avoids a square
    /// root operation.
    #[doc(alias = "magnitude2")]
    #[inline]
    #[must_use]
    pub fn length_squared(self) -> T {
        GVec2::from(self).length_squared()
    }

    /// Computes `1.0 / length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn length_recip(self) -> T {
        GVec2::from(self).length_recip()
    }

    /// Returns `self` normalized to length 1.0.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        Self::from_vec2(GVec2::from(self).normalize())
    }

    /// Returns `self` normalized to length `1.0` using a fast approximation.
    ///
    /// This can be faster than [`normalize()`], but is less accurate,
    /// and works best when `self` is already close to normalized.
    /// This is useful for preventing numerical drift when performing many
    /// successive multiplications of 2D rotations.
    ///
    /// [`normalize()`]: GRot2::normalize
    #[inline]
    #[must_use]
    pub fn normalize_fast(self) -> Self {
        // First-order Tayor approximation
        // 1/L = (L^2)^(-1/2) ≈ 1 - (L^2 - 1) / 2 = (3 - L^2) / 2
        let length_squared = self.length_squared();
        let approx_length_recip = T::HALF * (T::from_f32(3.0) - length_squared);
        Self::from_vec2(GVec2::from(self) * approx_length_recip)
    }

    /// Returns whether `self` is of length `1.0` or not.
    #[inline]
    #[must_use]
    pub fn is_normalized(self, eps: T) -> T::Bool {
        GVec2::from(self).is_normalized(eps)
    }

    /// Returns whether `self` is near the identity rotation.
    #[inline]
    #[must_use]
    pub fn is_near_identity(self) -> T::Bool {
        // Same as `GQuat::is_near_identity` but for 2D rotations.
        let threshold_sin = T::from_f32(0.000_049_692_047); // let threshold_angle = 0.002_847_144_6;
        self.cos.num_gt(T::ZERO) & self.sin.abs().num_lt(threshold_sin)
    }

    /// Returns the angle (in radians) for the minimal rotation
    /// for transforming this 2D rotation into another.
    ///
    /// Both rotations must be normalized.
    #[inline]
    #[must_use]
    pub fn angle_between(self, rhs: Self) -> T {
        // TODO: Should this take the absolute value?
        (rhs * self.inverse()).to_radians()
    }

    /// Rotates towards `rhs` up to `max_angle` (in radians).
    ///
    /// When `max_angle` is `0.0`, the result will be equal to `self`. When `max_angle` is equal to
    /// `self.angle_between(rhs)`, the result will be equal to `rhs`. If `max_angle` is negative,
    /// rotates towards the exact opposite of `rhs`. Will not go past the target.
    ///
    /// Both rotations must be normalized.
    #[inline]
    #[must_use]
    pub fn rotate_towards(self, rhs: Self, max_angle: T) -> Self {
        let angle = self.angle_between(rhs);
        let is_near = angle.num_le(T::from_f32(1e-4));
        let s = (max_angle / angle).clamp(T::NEG_ONE, T::ONE);
        Self::select(is_near, rhs, self.slerp(rhs, s))
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs`
    /// is less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two 2D rotations contain similar elements. It works
    /// best when comparing with a known value. The `max_abs_diff` that should be used used
    /// depends on the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: T) -> T::Bool {
        GVec2::from(self).abs_diff_eq(GVec2::from(rhs), max_abs_diff)
    }

    #[inline(always)]
    #[must_use]
    fn lerp_impl(self, end: Self, s: T) -> Self {
        (self * (T::ONE - s) + end * s).normalize()
    }

    /// Performs a linear interpolation between `self` and `rhs` based on
    /// the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.
    /// When `s` is `1.0`, the result will be equal to `rhs`.
    #[doc(alias = "mix")]
    #[inline]
    #[must_use]
    pub fn lerp(self, end: Self, s: T) -> Self {
        let dot = self.dot(end);
        let bias = dot.num_ge(T::ZERO).select(T::ONE, T::NEG_ONE);
        self.lerp_impl(end * bias, s)
    }

    /// Performs a spherical linear interpolation between `self` and `end`
    /// based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.
    /// When `s` is `1.0`, the result will be equal to `end`.
    #[inline]
    #[must_use]
    pub fn slerp(self, end: Self, s: T) -> Self {
        self * Self::from_radians(self.angle_between(end) * s)
    }

    /// Multiplies a quaternion and a 2D vector, returning the rotated vector.
    #[inline]
    #[must_use]
    pub fn mul_vec2(self, rhs: GVec2<T>) -> GVec2<T> {
        GVec2::new(
            self.cos * rhs.x - self.sin * rhs.y,
            self.sin * rhs.x + self.cos * rhs.y,
        )
    }

    /// Multiplies two 2D rotations. If they are both normalized, the result will
    /// represent the combined rotation.
    ///
    /// Note that due to floating point rounding, the result may not be perfectly normalized.
    /// Consider normalizing the result after several successive multiplications.
    #[inline]
    #[must_use]
    pub fn mul_rot2(self, rhs: Self) -> Self {
        Self::from_cos_sin(
            self.cos * rhs.cos - self.sin * rhs.sin,
            self.sin * rhs.cos + self.cos * rhs.sin,
        )
    }

    /// Adds an angle in radians to the rotation using a small-angle approximation,
    /// returning the resulting rotation.
    ///
    /// This can be faster than `self * GRot2::from_radians(angle)`, but is less accurate,
    /// and works best when `angle` is small. This is useful for cases like integrating
    /// angular velocity with small timesteps, where the rotation is periodically renormalized
    /// to prevent numerical drift.
    #[inline]
    #[must_use]
    pub fn add_radians_fast(self, angle: T) -> Self {
        let cos = self.cos - self.sin * angle;
        let sin = self.sin + self.cos * angle;
        Self::from_cos_sin(cos, sin).normalize_fast()
    }

    /// Creates a 2D rotation from a 2x2 rotation matrix inside a 2D affine transform.
    ///
    /// Note if the input affine matrix contain scales, shears, or other non-rotation transformations,
    /// the resulting rotation will be ill-defined.
    #[inline]
    #[must_use]
    pub fn from_affine2(a: &GAffine2<T>) -> Self {
        Self::from_mat2(&a.matrix2)
    }
}

/// # Float Methods
impl<T: Float> GRot2<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> T::Bool {
        GVec2::from(self).is_finite()
    }

    /// Returns `true` if any elements are `NAN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> T::Bool {
        GVec2::from(self).is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> GRot2<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar 2D rotation into a SIMD 2D rotation, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: GRot2<T::Element>) -> Self {
        Self::from_cos_sin(T::splat(value.cos), T::splat(value.sin))
    }

    /// Extracts the i-th lane of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub fn extract(&self, i: usize) -> GRot2<T::Element> {
        GRot2::from_cos_sin(self.cos.extract(i), self.sin.extract(i))
    }

    /// Extracts the i-th lane of `self` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub unsafe fn extract_unchecked(&self, i: usize) -> GRot2<T::Element> {
        unsafe { GRot2::from_cos_sin(self.cos.extract_unchecked(i), self.sin.extract_unchecked(i)) }
    }

    /// Replaces the i-th lane of `self` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    pub fn replace(&mut self, i: usize, value: GRot2<T::Element>) {
        self.cos.replace(i, value.cos);
        self.sin.replace(i, value.sin);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: GRot2<T::Element>) {
        unsafe {
            self.cos.replace_unchecked(i, value.cos);
            self.sin.replace_unchecked(i, value.sin);
        }
    }
}

impl<T: Real> Default for GRot2<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real + Add<Output = T>> Add for GRot2<T> {
    type Output = Self;
    /// Adds two 2D rotations.
    ///
    /// The sum is not guaranteed to be normalized.
    ///
    /// Note that addition is not the same as combining the rotations represented by the
    /// two rotations! That corresponds to multiplication.
    #[inline]
    fn add(self, rhs: GRot2<T>) -> Self {
        GRot2::from_vec2(GVec2::from(self) + GVec2::from(rhs))
    }
}

impl<T: Real + Add<Output = T>> Add<&GRot2<T>> for GRot2<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &GRot2<T>) -> Self {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<GRot2<T>> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn add(self, rhs: GRot2<T>) -> GRot2<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GRot2<T>> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn add(self, rhs: &GRot2<T>) -> GRot2<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> AddAssign for GRot2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: GRot2<T>) {
        *self = self.add(rhs);
    }
}

impl<T: Real + Add<Output = T>> AddAssign<&GRot2<T>> for GRot2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &GRot2<T>) {
        self.add_assign(*rhs);
    }
}

impl<T: Real + Sub<Output = T>> Sub for GRot2<T> {
    type Output = Self;
    /// Subtracts the `rhs` 2D rotation from `self`.
    ///
    /// The difference is not guaranteed to be normalized.
    ///
    /// Note that subtraction is not the same as combining the rotations represented by the
    /// two rotations! That corresponds to multiplication by the inverse.
    #[inline]
    fn sub(self, rhs: GRot2<T>) -> Self {
        GRot2::from_vec2(GVec2::from(self) - GVec2::from(rhs))
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GRot2<T>> for GRot2<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &GRot2<T>) -> Self {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<GRot2<T>> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn sub(self, rhs: GRot2<T>) -> GRot2<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GRot2<T>> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn sub(self, rhs: &GRot2<T>) -> GRot2<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> SubAssign for GRot2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: GRot2<T>) {
        *self = self.sub(rhs);
    }
}

impl<T: Real + Sub<Output = T>> SubAssign<&GRot2<T>> for GRot2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &GRot2<T>) {
        self.sub_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul for GRot2<T> {
    type Output = Self;
    /// Multiplies two 2D rotations. If they are both normalized, the result will
    /// represent the combined rotation.
    ///
    /// Note that due to floating point rounding, the result may not be perfectly normalized.
    /// Consider normalizing the result after several successive multiplications.
    #[inline]
    fn mul(self, rhs: GRot2<T>) -> Self {
        self.mul_rot2(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GRot2<T>> for GRot2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &GRot2<T>) -> Self {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GRot2<T>> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn mul(self, rhs: GRot2<T>) -> GRot2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GRot2<T>> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn mul(self, rhs: &GRot2<T>) -> GRot2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign for GRot2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GRot2<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&GRot2<T>> for GRot2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GRot2<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for GRot2<T> {
    type Output = GRot2<T>;
    /// Multiplies a 2D rotation by a scalar value.
    ///
    /// The product is not guaranteed to be normalized.
    #[inline]
    fn mul(self, rhs: T) -> GRot2<T> {
        Self::from_vec2(GVec2::from(self) * rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GRot2<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn mul(self, rhs: T) -> GRot2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GRot2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<T> for GRot2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&T> for GRot2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec2<T>> for GRot2<T> {
    type Output = GVec2<T>;
    /// Multiplies a 2D rotation and a 2D vector, returning the rotated vector.
    #[inline]
    fn mul(self, rhs: GVec2<T>) -> GVec2<T> {
        self.mul_vec2(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec2<T>> for GRot2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: &GVec2<T>) -> GVec2<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec2<T>> for &GRot2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec2<T>> for &GRot2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for GRot2<T> {
    type Output = Self;
    /// Divides a 2D rotation by a scalar value.
    ///
    /// The quotient is not guaranteed to be normalized.
    #[inline]
    fn div(self, rhs: T) -> Self {
        Self::from_vec2(GVec2::from(self) / rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for GRot2<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: &T) -> Self {
        self.div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn div(self, rhs: T) -> GRot2<T> {
        (*self).div(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn div(self, rhs: &T) -> GRot2<T> {
        (*self).div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> DivAssign<T> for GRot2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = self.div(rhs);
    }
}

impl<T: Real + Div<Output = T>> DivAssign<&T> for GRot2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.div_assign(*rhs);
    }
}

impl<T: Real> Neg for GRot2<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self * T::NEG_ONE
    }
}

impl<T: Real> Neg for &GRot2<T> {
    type Output = GRot2<T>;
    #[inline]
    fn neg(self) -> GRot2<T> {
        (*self).neg()
    }
}

impl<T: Real> Sum<GRot2<T>> for GRot2<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, T: Real> Sum<&'a GRot2<T>> for GRot2<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl<T: Real> Product<GRot2<T>> for GRot2<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl<'a, T: Real> Product<&'a GRot2<T>> for GRot2<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| a * b)
    }
}

impl<T: Real> From<GRot2<T>> for GVec2<T> {
    #[inline]
    fn from(v: GRot2<T>) -> Self {
        GVec2::new(v.cos, v.sin)
    }
}

impl<T: Real> From<[T; 2]> for GRot2<T> {
    #[inline]
    fn from(comps: [T; 2]) -> Self {
        Self::from_cos_sin(comps[0], comps[1])
    }
}

impl<T: Real> From<GRot2<T>> for [T; 2] {
    #[inline]
    fn from(v: GRot2<T>) -> Self {
        [v.cos, v.sin]
    }
}

impl<T: Real> From<(T, T)> for GRot2<T> {
    #[inline]
    fn from(comps: (T, T)) -> Self {
        Self::from_cos_sin(comps.0, comps.1)
    }
}

impl<T: Real> From<GRot2<T>> for (T, T) {
    #[inline]
    fn from(v: GRot2<T>) -> Self {
        (v.cos, v.sin)
    }
}

impl<T: Real> AsRef<[T; 2]> for GRot2<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 2] {
        unsafe { &*(self as *const GRot2<T> as *const [T; 2]) }
    }
}

impl<T: Real> AsMut<[T; 2]> for GRot2<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 2] {
        unsafe { &mut *(self as *mut GRot2<T> as *mut [T; 2]) }
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for GRot2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(p) = f.precision() {
            write!(f, "[{:.*}, {:.*}]", p, self.cos, p, self.sin)
        } else {
            write!(f, "[{}, {}]", self.cos, self.sin)
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for GRot2<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_tuple(stringify!(GRot2))
            .field(&self.cos)
            .field(&self.sin)
            .finish()
    }
}
