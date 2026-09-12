use crate::affine::GAffine2;
use crate::matrix::{GMat3, Mat2A};
use crate::rotation::GRot2;
use crate::vector::GVec2;

use core::{iter::Product, ops::*};

use gnum::num::{NumCast, Real};

/// A 2D affine transform with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Affine2`](crate::affine::Affine2),
/// at the cost of a larger size.
///
/// It can represent translation, rotation, scaling and shearing.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::AnyBitPattern))]
#[cfg_attr(
    feature = "zerocopy",
    derive(
        zerocopy_derive::FromBytes,
        zerocopy_derive::Immutable,
        zerocopy_derive::KnownLayout
    )
)]
#[repr(C)]
pub struct Affine2A {
    /// The 2x2 matrix containing the rotation and scale of the affine transformation.
    pub matrix2: Mat2A,
    /// The translation vector of the affine transformation.
    pub translation: GVec2<f32>,
}

/// # Constants
impl Affine2A {
    /// The degenerate zero transform.
    ///
    /// This transforms any finite vector and point to zero.
    /// The zero transform is non-invertible.
    pub const ZERO: Self = Self::from_cols(GVec2::ZERO, GVec2::ZERO, GVec2::ZERO);

    /// The identity transform.
    ///
    /// Multiplying a vector with this returns the same vector.
    pub const IDENTITY: Self = Self::from_cols(GVec2::X, GVec2::Y, GVec2::ZERO);

    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(GVec2::NAN, GVec2::NAN, GVec2::NAN);
}

/// # Construction
impl Affine2A {
    /// Creates an affine transform from three column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: GVec2<f32>, y_axis: GVec2<f32>, z_axis: GVec2<f32>) -> Self {
        Self {
            matrix2: Mat2A::from_cols(x_axis, y_axis),
            translation: z_axis,
        }
    }

    /// Creates an affine transform from a `[f32; 6]` array stored in column major order.
    #[inline]
    #[must_use]
    pub fn from_cols_array(m: &[f32; 6]) -> Self {
        Self {
            matrix2: Mat2A::from_cols_array(&[m[0], m[1], m[2], m[3]]),
            translation: GVec2::from_array([m[4], m[5]]),
        }
    }

    /// Creates a `[f32; 6]` array storing data in column major order.
    #[inline]
    #[must_use]
    pub fn to_cols_array(&self) -> [f32; 6] {
        let x = &self.matrix2.x_axis;
        let y = &self.matrix2.y_axis;
        let z = &self.translation;
        [x.x, x.y, y.x, y.y, z.x, z.y]
    }

    /// Creates an affine transform from a `[[f32; 2]; 3]` 2D array stored in column major order.
    ///
    /// If your data is in row major order you will need to `transpose` the returned matrix.
    #[inline]
    #[must_use]
    pub fn from_cols_array_2d(m: &[[f32; 2]; 3]) -> Self {
        Self {
            matrix2: Mat2A::from_cols(m[0].into(), m[1].into()),
            translation: m[2].into(),
        }
    }

    /// Creates a `[[f32; 2]; 3]` 2D array storing data in column major order.
    ///
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array_2d(&self) -> [[f32; 2]; 3] {
        [
            self.matrix2.x_axis.into(),
            self.matrix2.y_axis.into(),
            self.translation.into(),
        ]
    }

    /// Creates an affine transform from the first 6 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 6 elements long.
    #[inline]
    #[must_use]
    pub fn from_cols_slice(slice: &[f32]) -> Self {
        Self {
            matrix2: Mat2A::from_cols_slice(&slice[0..4]),
            translation: GVec2::from_slice(&slice[4..6]),
        }
    }

    /// Writes the columns of `self` to the first 6 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 6 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [f32]) {
        self.matrix2.write_cols_to_slice(&mut slice[0..4]);
        self.translation.write_to_slice(&mut slice[4..6]);
    }

    /// Creates an affine transformation from the elements in `if_true` and `if_false`,
    /// selecting which to use based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select(boolean: bool, if_true: Self, if_false: Self) -> Self {
        if boolean { if_true } else { if_false }
    }

    /// Creates an affine transform that changes scale.
    ///
    /// Note that if any scale is zero the transform will be non-invertible.
    #[inline]
    #[must_use]
    pub fn from_scale(scale: GVec2<f32>) -> Self {
        GAffine2::from_scale(scale).into()
    }

    /// Creates an affine transform from the given `rotation`.
    #[inline]
    #[must_use]
    #[doc(alias = "from_complex")]
    pub fn from_rotation(rotation: GRot2<f32>) -> Self {
        GAffine2::from_rotation(rotation).into()
    }

    /// Creates an affine transformation from the given 2D `translation`.
    #[inline]
    #[must_use]
    pub fn from_translation(translation: GVec2<f32>) -> Self {
        GAffine2::from_translation(translation).into()
    }

    /// Creates an affine transform from a 2x2 matrix (expressing scale, shear, and rotation).
    #[inline]
    #[must_use]
    pub fn from_mat2(matrix2: Mat2A) -> Self {
        Self {
            matrix2,
            translation: GVec2::ZERO,
        }
    }

    /// Creates an affine transform from a 2x2 matrix (expressing scale, shear, and rotation) and a
    /// translation vector.
    ///
    /// Equivalent to `Affine2A::from_translation(translation) * Affine2A::from_mat2(mat2)`.
    #[inline]
    #[must_use]
    pub fn from_mat2_translation(matrix2: Mat2A, translation: GVec2<f32>) -> Self {
        Self {
            matrix2,
            translation,
        }
    }

    /// Creates an affine transform from the given 2D `scale`, `rotation`, and `translation`.
    ///
    /// Equivalent to `Affine2A::from_translation(translation) *
    /// Affine2A::from_rotation(rotation) * Affine2A::from_scale(scale)`.
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: GVec2<f32>,
        rotation: GRot2<f32>,
        translation: GVec2<f32>,
    ) -> Self {
        GAffine2::from_scale_rotation_translation(scale, rotation, translation).into()
    }

    /// Creates an affine transform from the given 2D `rotation` and `translation`.
    ///
    /// Equivalent to `Affine2A::from_translation(translation) * Affine2A::from_rotation(rotation)`.
    #[inline]
    #[must_use]
    pub fn from_rotation_translation(rotation: GRot2<f32>, translation: GVec2<f32>) -> Self {
        GAffine2::from_rotation_translation(rotation, translation).into()
    }

    /// Creates an affine transform from a 3x3 matrix.
    ///
    /// The given matrix must be an affine transform and not contain any perspective transform.
    #[inline]
    #[must_use]
    pub fn from_mat3(m: GMat3<f32>) -> Self {
        GAffine2::from_mat3(m).into()
    }

    /// Extracts `scale`, `rotation` and `translation` from `self`.
    ///
    /// The transform is expected to be non-degenerate and without shearing, or the output
    /// will be invalid.
    #[inline]
    #[must_use]
    pub fn to_scale_rotation_translation(&self) -> (GVec2<f32>, GRot2<f32>, GVec2<f32>) {
        self.to_affine2().to_scale_rotation_translation()
    }
}

/// # Operations
impl Affine2A {
    /// Transforms the given 2D point, applying shear, scale, rotation and translation.
    #[inline]
    #[must_use]
    pub fn transform_point2(&self, rhs: GVec2<f32>) -> GVec2<f32> {
        self.matrix2 * rhs + self.translation
    }

    /// Transforms the given 2D vector, applying shear, scale and rotation (but NOT translation).
    ///
    /// To also apply translation, use [`Self::transform_point2()`] instead.
    #[inline]
    pub fn transform_vector2(&self, rhs: GVec2<f32>) -> GVec2<f32> {
        self.matrix2 * rhs
    }

    /// Return the inverse of this transform.
    ///
    /// Note that if the transform is not invertible the result will be invalid.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let matrix2 = self.matrix2.inverse();
        let translation = -(matrix2 * self.translation);

        Self {
            matrix2,
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
        self.matrix2.abs_diff_eq(rhs.matrix2, max_abs_diff)
            & self.translation.abs_diff_eq(rhs.translation, max_abs_diff)
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.matrix2.is_finite() & self.translation.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.matrix2.is_nan() | self.translation.is_nan()
    }
}

/// # Conversions
impl Affine2A {
    /// Creates an [`Affine2A`] from a [`GAffine2<f32>`].
    #[inline(always)]
    #[must_use]
    pub const fn from_affine2(a: GAffine2<f32>) -> Self {
        Self {
            matrix2: Mat2A::from_mat2(a.matrix2),
            translation: a.translation,
        }
    }

    /// Converts `self` to a [`GAffine2<f32>`].
    #[inline(always)]
    #[must_use]
    pub fn to_affine2(self) -> GAffine2<f32> {
        GAffine2 {
            matrix2: self.matrix2.to_mat2(),
            translation: self.translation,
        }
    }

    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GAffine2<U>
    where
        f32: NumCast<U>,
    {
        self.to_affine2().cast()
    }
}

impl Default for Affine2A {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Affine2A> for GMat3<f32> {
    #[inline]
    fn from(affine: Affine2A) -> Self {
        GMat3::from(affine.to_affine2())
    }
}

impl From<GAffine2<f32>> for Affine2A {
    #[inline(always)]
    fn from(a: GAffine2<f32>) -> Self {
        Self::from_affine2(a)
    }
}

impl From<Affine2A> for GAffine2<f32> {
    #[inline(always)]
    fn from(a: Affine2A) -> Self {
        a.to_affine2()
    }
}

impl Mul for Affine2A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            matrix2: self.matrix2 * rhs.matrix2,
            translation: self.matrix2 * rhs.translation + self.translation,
        }
    }
}

impl Mul<&Affine2A> for Affine2A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Affine2A) -> Self {
        self.mul(*rhs)
    }
}

impl MulAssign for Affine2A {
    #[inline]
    fn mul_assign(&mut self, rhs: Affine2A) {
        *self = self.mul(rhs);
    }
}

impl Mul<GMat3<f32>> for Affine2A {
    type Output = GMat3<f32>;
    #[inline]
    fn mul(self, rhs: GMat3<f32>) -> GMat3<f32> {
        GMat3::from(self) * rhs
    }
}

impl Mul<Affine2A> for GMat3<f32> {
    type Output = GMat3<f32>;
    #[inline]
    fn mul(self, rhs: Affine2A) -> GMat3<f32> {
        self * GMat3::from(rhs)
    }
}

impl Product<Affine2A> for Affine2A {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl Deref for Affine2A {
    type Target = crate::deref::Cols3<GVec2<f32>>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl DerefMut for Affine2A {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self as *mut Self::Target) }
    }
}

impl core::fmt::Display for Affine2A {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(p) = f.precision() {
            write!(
                f,
                "[{:.*}, {:.*}, {:.*}]",
                p, self.matrix2.x_axis, p, self.matrix2.y_axis, p, self.translation
            )
        } else {
            write!(
                f,
                "[{}, {}, {}]",
                self.matrix2.x_axis, self.matrix2.y_axis, self.translation
            )
        }
    }
}

impl core::fmt::Debug for Affine2A {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct("Affine2A")
            .field("matrix2", &self.matrix2)
            .field("translation", &self.translation)
            .finish()
    }
}
