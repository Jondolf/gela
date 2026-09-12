use crate::vector::GVec3;
use crate::{isometry::Iso3A, rotation::GRot3};

use core::{iter::Product, ops::*};

use gnum::{
    num::{Float, NumCast, Real},
    simd::Select,
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// A 3D isometry, which can represent rotation followed by translation.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "zerocopy", derive(FromBytes, Immutable, KnownLayout))]
#[repr(C)]
pub struct GIso3<T: Real> {
    /// The rotational part of the isometry.
    pub rotation: GRot3<T>,
    /// The translational part of the isometry.
    pub translation: GVec3<T>,
}

/// # Constants
impl<T: Real> GIso3<T> {
    /// The identity isometry.
    ///
    /// Multiplying a vector with this returns the same vector.
    pub const IDENTITY: Self = Self::from_rotation_translation(GRot3::IDENTITY, GVec3::ZERO);
}

impl<T: Float> GIso3<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::from_rotation_translation(GRot3::NAN, GVec3::NAN);
}

/// # Construction
impl<T: Real> GIso3<T> {
    /// Creates an isometry from a rotation and a translation.
    #[inline(always)]
    #[must_use]
    pub const fn from_rotation_translation(rotation: GRot3<T>, translation: GVec3<T>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Creates an isometry from the elements in `if_true` and `if_false`,
    /// selecting which to use based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select<B: Select<T>>(boolean: B, if_true: Self, if_false: Self) -> Self {
        Self {
            rotation: GRot3::select(boolean, if_true.rotation, if_false.rotation),
            translation: GVec3::select(boolean, if_true.translation, if_false.translation),
        }
    }

    /// Creates an isometry from the given `rotation`, with no translation.
    #[inline]
    #[must_use]
    pub fn from_rotation(rotation: GRot3<T>) -> Self {
        Self {
            rotation,
            translation: GVec3::ZERO,
        }
    }

    /// Creates an isometry from the given 3D `translation`, with no rotation.
    #[inline]
    #[must_use]
    pub fn from_translation(translation: GVec3<T>) -> Self {
        Self {
            rotation: GRot3::IDENTITY,
            translation,
        }
    }
}

/// # Operations
impl<T: Real> GIso3<T> {
    /// Transforms the given 3D point, applying rotation and translation.
    #[inline]
    #[must_use]
    pub fn transform_point3(&self, rhs: GVec3<T>) -> GVec3<T> {
        self.rotation * rhs + self.translation
    }

    /// Transforms the given 3D vector, applying rotation (but NOT translation).
    ///
    /// To also apply translation, use [`Self::transform_point3()`] instead.
    #[inline]
    pub fn transform_vector3(&self, rhs: GVec3<T>) -> GVec3<T> {
        self.rotation * rhs
    }

    /// Return the inverse of this transform.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        let translation = -(rotation * self.translation);

        Self {
            rotation,
            translation,
        }
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs`
    /// is less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two 3x4 matrices contain similar elements. It works
    /// best when comparing with a known value. The `max_abs_diff` that should be used used
    /// depends on the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: T) -> T::Bool {
        self.rotation.abs_diff_eq(rhs.rotation, max_abs_diff)
            & self.translation.abs_diff_eq(rhs.translation, max_abs_diff)
    }
}

impl<T: Float> GIso3<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> T::Bool {
        self.rotation.is_finite() & self.translation.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> T::Bool {
        self.rotation.is_nan() | self.translation.is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> GIso3<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar isometry into a SIMD isometry, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: GIso3<T::Element>) -> Self {
        Self {
            rotation: GRot3::broadcast(value.rotation),
            translation: GVec3::broadcast(value.translation),
        }
    }

    /// Extracts the i-th lane of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub fn extract(&self, i: usize) -> GIso3<T::Element> {
        GIso3 {
            rotation: self.rotation.extract(i),
            translation: self.translation.extract(i),
        }
    }

    /// Extracts the i-th lane of `self` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub unsafe fn extract_unchecked(&self, i: usize) -> GIso3<T::Element> {
        unsafe {
            GIso3 {
                rotation: self.rotation.extract_unchecked(i),
                translation: self.translation.extract_unchecked(i),
            }
        }
    }

    /// Replaces the i-th lane of `self` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    pub fn replace(&mut self, i: usize, value: GIso3<T::Element>) {
        self.rotation.replace(i, value.rotation);
        self.translation.replace(i, value.translation);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: GIso3<T::Element>) {
        unsafe {
            self.rotation.replace_unchecked(i, value.rotation);
            self.translation.replace_unchecked(i, value.translation);
        }
    }
}

/// # Conversion
impl<T: Real> GIso3<T> {
    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GIso3<U>
    where
        T: NumCast<U>,
    {
        GIso3::from_rotation_translation(self.rotation.cast(), self.translation.cast())
    }
}

impl GIso3<f32> {
    /// Converts `self` to an [`Iso3A`].
    #[inline(always)]
    #[must_use]
    pub const fn to_iso3a(self) -> Iso3A {
        Iso3A::from_iso3(self)
    }
}

impl GIso3<f64> {
    /// Converts `self` to an [`Iso3A`].
    #[inline]
    #[must_use]
    pub fn to_iso3a(self) -> Iso3A {
        Iso3A::from_iso3(self.cast())
    }
}

impl<T: Real> Default for GIso3<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real + Mul<Output = T>> Mul for GIso3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: GIso3<T>) -> Self {
        Self {
            rotation: self.rotation * rhs.rotation,
            translation: self.rotation * rhs.translation + self.translation,
        }
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GIso3<T>> for GIso3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &GIso3<T>) -> Self {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GIso3<T>> for &GIso3<T> {
    type Output = GIso3<T>;
    #[inline]
    fn mul(self, rhs: GIso3<T>) -> GIso3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GIso3<T>> for &GIso3<T> {
    type Output = GIso3<T>;
    #[inline]
    fn mul(self, rhs: &GIso3<T>) -> GIso3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign for GIso3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GIso3<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + MulAssign> MulAssign<&GIso3<T>> for GIso3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GIso3<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec3<T>> for GIso3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: GVec3<T>) -> GVec3<T> {
        self.transform_point3(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec3<T>> for GIso3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: &GVec3<T>) -> GVec3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec3<T>> for &GIso3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: GVec3<T>) -> GVec3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec3<T>> for &GIso3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: &GVec3<T>) -> GVec3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real> Product<GIso3<T>> for GIso3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl<'a, T: Real> Product<&'a GIso3<T>> for GIso3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| a * b)
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for GIso3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(p) = f.precision() {
            write!(
                f,
                "GIso3 {{ rotation: {:.*}, translation: {:.*} }}",
                p, self.rotation, p, self.translation
            )
        } else {
            write!(
                f,
                "GIso3 {{ rotation: {}, translation: {} }}",
                self.rotation, self.translation
            )
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for GIso3<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(GIso3))
            .field("rotation", &self.rotation)
            .field("translation", &self.translation)
            .finish()
    }
}
