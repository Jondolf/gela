use crate::isometry::GIso3;
use crate::rotation::Rot3A;
use crate::vector::{GVec3, Vec3A};

use core::{iter::Product, ops::*};

use gnum::num::{NumCast, Real};

/// A 3D isometry with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Iso3`](crate::isometry::Iso3).
///
/// It can represent rotation followed by translation.
#[derive(Clone, Copy, PartialEq)]
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
#[repr(C)]
pub struct Iso3A {
    /// The rotational part of the isometry.
    pub rotation: Rot3A,
    /// The translational part of the isometry.
    pub translation: Vec3A,
}

/// # Constants
impl Iso3A {
    /// The identity isometry.
    ///
    /// Multiplying a vector with this returns the same vector.
    pub const IDENTITY: Self = Self::from_rotation_translation(Rot3A::IDENTITY, Vec3A::ZERO);

    /// All `NAN`.
    pub const NAN: Self = Self::from_rotation_translation(Rot3A::NAN, Vec3A::NAN);
}

/// # Construction
impl Iso3A {
    /// Creates an isometry from a rotation and a translation.
    #[inline(always)]
    #[must_use]
    pub const fn from_rotation_translation(rotation: Rot3A, translation: Vec3A) -> Self {
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
    pub fn select(boolean: bool, if_true: Self, if_false: Self) -> Self {
        if boolean { if_true } else { if_false }
    }

    /// Creates an isometry from the given `rotation`, with no translation.
    #[inline]
    #[must_use]
    pub fn from_rotation(rotation: Rot3A) -> Self {
        Self {
            rotation,
            translation: Vec3A::ZERO,
        }
    }

    /// Creates an isometry from the given 3D `translation`, with no rotation.
    #[inline]
    #[must_use]
    pub fn from_translation(translation: Vec3A) -> Self {
        Self {
            rotation: Rot3A::IDENTITY,
            translation,
        }
    }
}

/// # Operations
impl Iso3A {
    /// Transforms the given 3D point, applying rotation and translation.
    #[inline]
    #[must_use]
    pub fn transform_point3(&self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.rotation.mul_vec3(rhs) + self.translation.to_vec3()
    }

    /// Transforms the given SIMD-aligned 3D point, applying rotation and translation.
    #[inline]
    #[must_use]
    pub fn transform_point3a(&self, rhs: Vec3A) -> Vec3A {
        self.rotation.mul_vec3a(rhs) + self.translation
    }

    /// Transforms the given 3D vector, applying rotation (but NOT translation).
    ///
    /// To also apply translation, use [`Self::transform_point3()`] instead.
    #[inline]
    pub fn transform_vector3(&self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.rotation.mul_vec3(rhs)
    }

    /// Transforms the given SIMD-aligned 3D vector, applying rotation (but NOT translation).
    ///
    /// To also apply translation, use [`Self::transform_point3a()`] instead.
    #[inline]
    #[must_use]
    pub fn transform_vector3a(&self, rhs: Vec3A) -> Vec3A {
        self.rotation.mul_vec3a(rhs)
    }

    /// Return the inverse of this transform.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        let translation = -rotation.mul_vec3a(self.translation);

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
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: f32) -> bool {
        self.rotation.abs_diff_eq(rhs.rotation, max_abs_diff)
            & self.translation.abs_diff_eq(rhs.translation, max_abs_diff)
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.rotation.is_finite() & self.translation.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.rotation.is_nan() | self.translation.is_nan()
    }
}

/// # Conversions
impl Iso3A {
    /// Creates an [`Iso3A`] from a [`GIso3<f32>`].
    #[inline(always)]
    #[must_use]
    pub const fn from_iso3(i: GIso3<f32>) -> Self {
        Self {
            rotation: Rot3A::from_quat(i.rotation),
            translation: Vec3A::from_vec3(i.translation),
        }
    }

    /// Converts `self` to a [`GIso3<f32>`].
    #[inline(always)]
    #[must_use]
    pub fn to_iso3(self) -> GIso3<f32> {
        GIso3 {
            rotation: self.rotation.to_quat(),
            translation: self.translation.to_vec3(),
        }
    }

    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GIso3<U>
    where
        f32: NumCast<U>,
    {
        self.to_iso3().cast()
    }
}

impl Default for Iso3A {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<GIso3<f32>> for Iso3A {
    #[inline(always)]
    fn from(i: GIso3<f32>) -> Self {
        Self::from_iso3(i)
    }
}

impl From<Iso3A> for GIso3<f32> {
    #[inline(always)]
    fn from(i: Iso3A) -> Self {
        i.to_iso3()
    }
}

impl Mul for Iso3A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Iso3A) -> Self {
        Self {
            rotation: self.rotation * rhs.rotation,
            translation: self.rotation.mul_vec3a(rhs.translation) + self.translation,
        }
    }
}

impl Mul<&Iso3A> for Iso3A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Iso3A) -> Self {
        self.mul(*rhs)
    }
}

impl MulAssign for Iso3A {
    #[inline]
    fn mul_assign(&mut self, rhs: Iso3A) {
        *self = self.mul(rhs);
    }
}

impl Product<Iso3A> for Iso3A {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl core::fmt::Display for Iso3A {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(p) = f.precision() {
            write!(
                f,
                "Iso3A {{ rotation: {:.*}, translation: {:.*} }}",
                p, self.rotation, p, self.translation
            )
        } else {
            write!(
                f,
                "Iso3A {{ rotation: {}, translation: {} }}",
                self.rotation, self.translation
            )
        }
    }
}

impl core::fmt::Debug for Iso3A {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct("Iso3A")
            .field("rotation", &self.rotation)
            .field("translation", &self.translation)
            .finish()
    }
}
