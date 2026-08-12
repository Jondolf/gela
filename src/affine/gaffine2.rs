use crate::matrix::{GMat2, GMat3};
use crate::rotation::GRot2;
use crate::vector::{GVec2, Vec3Swizzles};

use core::{iter::Product, ops::*};

use gnum::{
    num::{Float, NumCast, Real},
    simd::Select,
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// A 2D affine transform, which can represent translation, rotation, scaling and shearing.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::AnyBitPattern))]
#[cfg_attr(feature = "zerocopy", derive(FromBytes, Immutable, KnownLayout))]
#[repr(C)]
pub struct GAffine2<T: Real> {
    /// The 2x2 matrix containing the rotation and scale of the affine transformation.
    pub matrix2: GMat2<T>,
    /// The translation vector of the affine transformation.
    pub translation: GVec2<T>,
}

/// # Constants
impl<T: Real> GAffine2<T> {
    /// The degenerate zero transform.
    ///
    /// This transforms any finite vector and point to zero.
    /// The zero transform is non-invertible.
    pub const ZERO: Self = Self::from_cols(GVec2::ZERO, GVec2::ZERO, GVec2::ZERO);

    /// The identity transform.
    ///
    /// Multiplying a vector with this returns the same vector.
    pub const IDENTITY: Self = Self::from_cols(GVec2::X, GVec2::Y, GVec2::ZERO);
}

impl<T: Float> GAffine2<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(GVec2::NAN, GVec2::NAN, GVec2::NAN);
}

/// # Construction
impl<T: Real> GAffine2<T> {
    /// Creates an affine transform from three column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: GVec2<T>, y_axis: GVec2<T>, z_axis: GVec2<T>) -> Self {
        Self {
            matrix2: GMat2::from_cols(x_axis, y_axis),
            translation: z_axis,
        }
    }

    /// Creates an affine transform from a `[T; 6]` array stored in column major order.
    #[inline]
    #[must_use]
    pub fn from_cols_array(m: &[T; 6]) -> Self {
        Self {
            matrix2: GMat2::from_cols_array(&[m[0], m[1], m[2], m[3]]),
            translation: GVec2::from_array([m[4], m[5]]),
        }
    }

    /// Creates a `[T; 6]` array storing data in column major order.
    #[inline]
    #[must_use]
    pub fn to_cols_array(&self) -> [T; 6] {
        let x = &self.matrix2.x_axis;
        let y = &self.matrix2.y_axis;
        let z = &self.translation;
        [x.x, x.y, y.x, y.y, z.x, z.y]
    }

    /// Creates an affine transform from a `[[T; 2]; 3]` 2D array stored in column major order.
    ///
    /// If your data is in row major order you will need to `transpose` the returned matrix.
    #[inline]
    #[must_use]
    pub fn from_cols_array_2d(m: &[[T; 2]; 3]) -> Self {
        Self {
            matrix2: GMat2::from_cols(m[0].into(), m[1].into()),
            translation: m[2].into(),
        }
    }

    /// Creates a `[[T; 2]; 3]` 2D array storing data in column major order.
    ///
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array_2d(&self) -> [[T; 2]; 3] {
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
    pub fn from_cols_slice(slice: &[T]) -> Self {
        Self {
            matrix2: GMat2::from_cols_slice(&slice[0..4]),
            translation: GVec2::from_slice(&slice[4..6]),
        }
    }

    /// Writes the columns of `self` to the first 6 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 6 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [T]) {
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
    pub fn select<B: Select<T>>(boolean: B, if_true: Self, if_false: Self) -> Self {
        Self {
            matrix2: GMat2::select(boolean, if_true.matrix2, if_false.matrix2),
            translation: GVec2::select(boolean, if_true.translation, if_false.translation),
        }
    }

    /// Creates an affine transform that changes scale.
    ///
    /// Note that if any scale is zero the transform will be non-invertible.
    #[inline]
    #[must_use]
    pub fn from_scale(scale: GVec2<T>) -> Self {
        Self {
            matrix2: GMat2::from_diagonal(scale),
            translation: GVec2::ZERO,
        }
    }

    /// Creates an affine transform from the given `rotation`.
    #[inline]
    #[must_use]
    #[doc(alias = "from_complex")]
    pub fn from_rotation(rotation: GRot2<T>) -> Self {
        Self {
            matrix2: GMat2::from_rotation(rotation),
            translation: GVec2::ZERO,
        }
    }

    /// Creates an affine transformation from the given 2D `translation`.
    #[inline]
    #[must_use]
    pub fn from_translation(translation: GVec2<T>) -> Self {
        Self {
            matrix2: GMat2::IDENTITY,
            translation,
        }
    }

    /// Creates an affine transform from a 2x2 matrix (expressing scale, shear, and rotation).
    #[inline]
    #[must_use]
    pub fn from_mat2(matrix2: GMat2<T>) -> Self {
        Self {
            matrix2,
            translation: GVec2::ZERO,
        }
    }

    /// Creates an affine transform from a 2x2 matrix (expressing scale, shear, and rotation) and a
    /// translation vector.
    ///
    /// Equivalent to `GAffine2::from_translation(translation) * GAffine2::from_mat2(mat2)`.
    #[inline]
    #[must_use]
    pub fn from_mat2_translation(matrix2: GMat2<T>, translation: GVec2<T>) -> Self {
        Self {
            matrix2,
            translation,
        }
    }

    /// Creates an affine transform from the given 2D `scale`, `rotation`, and `translation`.
    ///
    /// Equivalent to `GAffine2::from_translation(translation) *
    /// GAffine2::from_rotation(rotation) * GAffine2::from_scale(scale)`.
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: GVec2<T>,
        rotation: GRot2<T>,
        translation: GVec2<T>,
    ) -> Self {
        let rotation = GMat2::from_rotation(rotation);
        Self {
            matrix2: GMat2::from_cols(rotation.x_axis * scale.x, rotation.y_axis * scale.y),
            translation,
        }
    }

    /// Creates an affine transform from the given 2D `rotation` and `translation`.
    ///
    /// Equivalent to `GAffine2::from_translation(translation) * GAffine2::from_rotation(rotation)`.
    #[inline]
    #[must_use]
    pub fn from_rotation_translation(rotation: GRot2<T>, translation: GVec2<T>) -> Self {
        Self {
            matrix2: GMat2::from_rotation(rotation),
            translation,
        }
    }

    /// Creates an affine transform from a 3x3 matrix.
    ///
    /// The given matrix must be an affine transform and not contain any perspective transform.
    #[inline]
    #[must_use]
    pub fn from_mat3(m: GMat3<T>) -> Self {
        Self {
            matrix2: GMat2::from_cols(m.x_axis.xy(), m.y_axis.xy()),
            translation: m.z_axis.xy(),
        }
    }

    /// Extracts `scale`, `rotation` and `translation` from `self`.
    ///
    /// The transform is expected to be non-degenerate and without shearing, or the output
    /// will be invalid.
    #[inline]
    #[must_use]
    pub fn to_scale_rotation_translation(&self) -> (GVec2<T>, GRot2<T>, GVec2<T>) {
        let det = self.matrix2.determinant();

        let scale = GVec2::new(
            self.matrix2.x_axis.length() * det.signum(),
            self.matrix2.y_axis.length(),
        );

        let cos = self.matrix2.x_axis.x / scale.x;
        let sin = self.matrix2.x_axis.y / scale.x;
        let rotation = GRot2::from_cos_sin(cos, sin);

        (scale, rotation, self.translation)
    }
}

/// # Operations
impl<T: Real> GAffine2<T> {
    /// Transforms the given 2D point, applying shear, scale, rotation and translation.
    #[inline]
    #[must_use]
    pub fn transform_point2(&self, rhs: GVec2<T>) -> GVec2<T> {
        self.matrix2 * rhs + self.translation
    }

    /// Transforms the given 2D vector, applying shear, scale and rotation (but NOT translation).
    ///
    /// To also apply translation, use [`Self::transform_point2()`] instead.
    #[inline]
    pub fn transform_vector2(&self, rhs: GVec2<T>) -> GVec2<T> {
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
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: T) -> T::Bool {
        self.matrix2.abs_diff_eq(rhs.matrix2, max_abs_diff)
            & self.translation.abs_diff_eq(rhs.translation, max_abs_diff)
    }
}

impl<T: Float> GAffine2<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> T::Bool {
        self.matrix2.is_finite() & self.translation.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> T::Bool {
        self.matrix2.is_nan() | self.translation.is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> GAffine2<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar affine into a SIMD affine, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: GAffine2<T::Element>) -> Self {
        Self {
            matrix2: GMat2::broadcast(value.matrix2),
            translation: GVec2::broadcast(value.translation),
        }
    }

    /// Extracts the i-th lane of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub fn extract(&self, i: usize) -> GAffine2<T::Element> {
        GAffine2 {
            matrix2: self.matrix2.extract(i),
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
    pub unsafe fn extract_unchecked(&self, i: usize) -> GAffine2<T::Element> {
        unsafe {
            GAffine2 {
                matrix2: self.matrix2.extract_unchecked(i),
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
    pub fn replace(&mut self, i: usize, value: GAffine2<T::Element>) {
        self.matrix2.replace(i, value.matrix2);
        self.translation.replace(i, value.translation);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: GAffine2<T::Element>) {
        unsafe {
            self.matrix2.replace_unchecked(i, value.matrix2);
            self.translation.replace_unchecked(i, value.translation);
        }
    }
}

/// # Conversion
impl<T: Real> GAffine2<T> {
    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GAffine2<U>
    where
        T: NumCast<U>,
    {
        GAffine2 {
            matrix2: self.matrix2.cast(),
            translation: self.translation.cast(),
        }
    }
}

impl<T: Real> Default for GAffine2<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real> From<GAffine2<T>> for GMat3<T> {
    #[inline]
    fn from(affine: GAffine2<T>) -> Self {
        Self::from_cols(
            affine.matrix2.x_axis.extend(T::ZERO),
            affine.matrix2.y_axis.extend(T::ZERO),
            affine.translation.extend(T::ONE),
        )
    }
}

impl<T: Real + Mul<Output = T>> Mul for GAffine2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: GAffine2<T>) -> Self {
        Self {
            matrix2: self.matrix2 * rhs.matrix2,
            translation: self.matrix2 * rhs.translation + self.translation,
        }
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GAffine2<T>> for GAffine2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &GAffine2<T>) -> Self {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GAffine2<T>> for &GAffine2<T> {
    type Output = GAffine2<T>;
    #[inline]
    fn mul(self, rhs: GAffine2<T>) -> GAffine2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GAffine2<T>> for &GAffine2<T> {
    type Output = GAffine2<T>;
    #[inline]
    fn mul(self, rhs: &GAffine2<T>) -> GAffine2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign for GAffine2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GAffine2<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + MulAssign> MulAssign<&GAffine2<T>> for GAffine2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GAffine2<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat3<T>> for GAffine2<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: GMat3<T>) -> GMat3<T> {
        GMat3::from(self) * rhs
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat3<T>> for GAffine2<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &GMat3<T>) -> GMat3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat3<T>> for &GAffine2<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: GMat3<T>) -> GMat3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat3<T>> for &GAffine2<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &GMat3<T>) -> GMat3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GAffine2<T>> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: GAffine2<T>) -> GMat3<T> {
        self * GMat3::from(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GAffine2<T>> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &GAffine2<T>) -> GMat3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GAffine2<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: GAffine2<T>) -> GMat3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GAffine2<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &GAffine2<T>) -> GMat3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<GAffine2<T>> for GMat3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GAffine2<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&GAffine2<T>> for GMat3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GAffine2<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real> Product<GAffine2<T>> for GAffine2<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl<'a, T: Real> Product<&'a GAffine2<T>> for GAffine2<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| a * b)
    }
}

impl<T: Real> Deref for GAffine2<T> {
    type Target = crate::deref::Cols3<GVec2<T>>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl<T: Real> DerefMut for GAffine2<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self as *mut Self::Target) }
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for GAffine2<T> {
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

impl<T: Real + core::fmt::Debug> core::fmt::Debug for GAffine2<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(GAffine2))
            .field("matrix2", &self.matrix2)
            .field("translation", &self.translation)
            .finish()
    }
}
