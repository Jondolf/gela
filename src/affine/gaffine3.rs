use crate::matrix::{GMat3, GMat4};
use crate::rotation::GRot3;
use crate::vector::{GVec3, Vec4Swizzles};

use core::{iter::Product, ops::*};

use gnum::{
    num::{Float, NumCast, Real},
    simd::Select,
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// A 3D affine transform, which can represent translation, rotation, scaling and shearing.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "zerocopy", derive(FromBytes, Immutable, KnownLayout))]
#[repr(C)]
pub struct GAffine3<T: Real> {
    /// The 3x3 matrix containing the rotation and scale of the affine transformation.
    pub matrix3: GMat3<T>,
    /// The translation vector of the affine transformation.
    pub translation: GVec3<T>,
}

/// # Constants
impl<T: Real> GAffine3<T> {
    /// The degenerate zero transform.
    ///
    /// This transforms any finite vector and point to zero.
    /// The zero transform is non-invertible.
    pub const ZERO: Self = Self::from_cols(GVec3::ZERO, GVec3::ZERO, GVec3::ZERO, GVec3::ZERO);

    /// The identity transform.
    ///
    /// Multiplying a vector with this returns the same vector.
    pub const IDENTITY: Self = Self::from_cols(GVec3::X, GVec3::Y, GVec3::Z, GVec3::ZERO);
}

impl<T: Float> GAffine3<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(GVec3::NAN, GVec3::NAN, GVec3::NAN, GVec3::NAN);
}

/// # Construction
impl<T: Real> GAffine3<T> {
    /// Creates an affine transform from four column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(
        x_axis: GVec3<T>,
        y_axis: GVec3<T>,
        z_axis: GVec3<T>,
        w_axis: GVec3<T>,
    ) -> Self {
        Self {
            matrix3: GMat3::from_cols(x_axis, y_axis, z_axis),
            translation: w_axis,
        }
    }

    /// Creates an affine transform from a `[T; 12]` array stored in column major order.
    #[inline]
    #[must_use]
    pub fn from_cols_array(m: &[T; 12]) -> Self {
        Self {
            matrix3: GMat3::from_cols_array(&[
                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
            ]),
            translation: GVec3::from_array([m[9], m[10], m[11]]),
        }
    }

    /// Creates a `[T; 12]` array storing data in column major order.
    #[inline]
    #[must_use]
    pub fn to_cols_array(&self) -> [T; 12] {
        let x = &self.matrix3.x_axis;
        let y = &self.matrix3.y_axis;
        let z = &self.matrix3.z_axis;
        let w = &self.translation;
        [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z, w.x, w.y, w.z]
    }

    /// Creates an affine transform from a `[[T; 3]; 4]` 2D array stored in column major order.
    ///
    /// If your data is in row major order you will need to `transpose` the returned matrix.
    #[inline]
    #[must_use]
    pub fn from_cols_array_2d(m: &[[T; 3]; 4]) -> Self {
        Self {
            matrix3: GMat3::from_cols(m[0].into(), m[1].into(), m[2].into()),
            translation: m[3].into(),
        }
    }

    /// Creates a `[[T; 3]; 4]` 2D array storing data in column major order.
    ///
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array_2d(&self) -> [[T; 3]; 4] {
        [
            self.matrix3.x_axis.into(),
            self.matrix3.y_axis.into(),
            self.matrix3.z_axis.into(),
            self.translation.into(),
        ]
    }

    /// Creates an affine transform from the first 12 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 12 elements long.
    #[inline]
    #[must_use]
    pub fn from_cols_slice(slice: &[T]) -> Self {
        Self {
            matrix3: GMat3::from_cols_slice(&slice[0..9]),
            translation: GVec3::from_slice(&slice[9..12]),
        }
    }

    /// Writes the columns of `self` to the first 12 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 12 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [T]) {
        self.matrix3.write_cols_to_slice(&mut slice[0..9]);
        self.translation.write_to_slice(&mut slice[9..12]);
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
            matrix3: GMat3::select(boolean, if_true.matrix3, if_false.matrix3),
            translation: GVec3::select(boolean, if_true.translation, if_false.translation),
        }
    }

    /// Creates an affine transform that changes scale.
    ///
    /// Note that if any scale is zero the transform will be non-invertible.
    #[inline]
    #[must_use]
    pub fn from_scale(scale: GVec3<T>) -> Self {
        Self {
            matrix3: GMat3::from_diagonal(scale),
            translation: GVec3::ZERO,
        }
    }

    /// Creates an affine transform from the given `rotation`.
    #[inline]
    #[must_use]
    #[doc(alias = "from_quat")]
    pub fn from_rotation(rotation: GRot3<T>) -> Self {
        Self {
            matrix3: GMat3::from_rotation(rotation),
            translation: GVec3::ZERO,
        }
    }

    /// Creates an affine transform containing a 3D rotation around a normalized
    /// rotation `axis` of `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: GVec3<T>, angle: T) -> Self {
        Self {
            matrix3: GMat3::from_axis_angle(axis, angle),
            translation: GVec3::ZERO,
        }
    }

    /// Creates an affine transform containing a 3D rotation around the x axis of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: T) -> Self {
        Self {
            matrix3: GMat3::from_rotation_x(angle),
            translation: GVec3::ZERO,
        }
    }

    /// Creates an affine transform containing a 3D rotation around the y axis of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: T) -> Self {
        Self {
            matrix3: GMat3::from_rotation_y(angle),
            translation: GVec3::ZERO,
        }
    }

    /// Creates an affine transform containing a 3D rotation around the z axis of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: T) -> Self {
        Self {
            matrix3: GMat3::from_rotation_z(angle),
            translation: GVec3::ZERO,
        }
    }

    /// Creates an affine transformation from the given 3D `translation`.
    #[inline]
    #[must_use]
    pub fn from_translation(translation: GVec3<T>) -> Self {
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: GMat3::IDENTITY,
            translation: translation.into(),
        }
    }

    /// Creates an affine transform from a 3x3 matrix (expressing scale, shear, and rotation).
    #[inline]
    #[must_use]
    pub fn from_mat3(mat3: GMat3<T>) -> Self {
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: mat3.into(),
            translation: GVec3::ZERO,
        }
    }

    /// Creates an affine transform from a 3x3 matrix (expressing scale, shear, and rotation)
    /// and a translation vector.
    ///
    /// Equivalent to `GAffine3::from_translation(translation) * GAffine3::from_mat3(mat3)`
    #[inline]
    #[must_use]
    pub fn from_mat3_translation(mat3: GMat3<T>, translation: GVec3<T>) -> Self {
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: mat3.into(),
            translation: translation.into(),
        }
    }

    /// Creates an affine transform from the given 3D `scale`, `rotation`, and `translation`.
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: GVec3<T>,
        rotation: GRot3<T>,
        translation: GVec3<T>,
    ) -> Self {
        let rotation = GMat3::from_rotation(rotation);
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: GMat3::from_cols(
                rotation.x_axis * scale.x,
                rotation.y_axis * scale.y,
                rotation.z_axis * scale.z,
            ),
            translation: translation.into(),
        }
    }

    /// Creates an affine transform from the given 3D `rotation` and `translation`.
    #[inline]
    #[must_use]
    pub fn from_rotation_translation(rotation: GRot3<T>, translation: GVec3<T>) -> Self {
        #[allow(clippy::useless_conversion)]
        Self {
            matrix3: GMat3::from_rotation(rotation),
            translation: translation.into(),
        }
    }

    /// Creates an affine transform from a 4x4 matrix.
    ///
    /// The given matrix must be an affine transform and not contain any perspective transform.
    #[inline]
    #[must_use]
    pub fn from_mat4(m: GMat4<T>) -> Self {
        Self {
            matrix3: GMat3::from_cols(m.x_axis.xyz(), m.y_axis.xyz(), m.z_axis.xyz()),
            translation: m.w_axis.xyz(),
        }
    }

    /// Extracts `scale`, `rotation` and `translation` from `self`.
    ///
    /// The transform is expected to be non-degenerate and without shearing, or the output
    /// will be invalid.
    #[inline]
    #[must_use]
    pub fn to_scale_rotation_translation(&self) -> (GVec3<T>, GRot3<T>, GVec3<T>) {
        let det = self.matrix3.determinant();

        let scale = GVec3::new(
            self.matrix3.x_axis.length() * det.signum(),
            self.matrix3.y_axis.length(),
            self.matrix3.z_axis.length(),
        );

        let inv_scale = scale.recip();

        #[allow(clippy::useless_conversion)]
        let rotation = GRot3::from_mat3(&GMat3::from_cols(
            (self.matrix3.x_axis * inv_scale.x).into(),
            (self.matrix3.y_axis * inv_scale.y).into(),
            (self.matrix3.z_axis * inv_scale.z).into(),
        ));

        #[allow(clippy::useless_conversion)]
        (scale, rotation, self.translation.into())
    }

    /// Creates a left-handed view transform using a camera position, an up direction, and a facing direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(eye: GVec3<T>, dir: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_rh(eye, -dir, up)
    }

    /// Creates a right-handed view transform using a camera position, an up direction, and a facing direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(eye: GVec3<T>, dir: GVec3<T>, up: GVec3<T>) -> Self {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self {
            matrix3: GMat3::from_cols(
                GVec3::new(s.x, u.x, -f.x),
                GVec3::new(s.y, u.y, -f.y),
                GVec3::new(s.z, u.z, -f.z),
            ),
            translation: GVec3::new(-eye.dot(s), -eye.dot(u), eye.dot(f)),
        }
    }

    /// Creates a left-handed view transform using a camera position, an up direction, and a focal
    /// point. For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: GVec3<T>, center: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_lh(eye, center - eye, up)
    }

    /// Creates a right-handed view transform using a camera position, an up direction, and a focal
    /// point. For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_at_rh(eye: GVec3<T>, center: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_rh(eye, center - eye, up)
    }
}

/// # Operations
impl<T: Real> GAffine3<T> {
    /// Transforms the given 3D points, applying shear, scale, rotation and translation.
    #[inline]
    pub fn transform_point3(&self, rhs: GVec3<T>) -> GVec3<T> {
        #[allow(clippy::useless_conversion)]
        ((self.matrix3.x_axis * rhs.x)
            + (self.matrix3.y_axis * rhs.y)
            + (self.matrix3.z_axis * rhs.z)
            + self.translation)
            .into()
    }

    /// Transforms the given 3D vector, applying shear, scale and rotation (but NOT translation).
    ///
    /// To also apply translation, use [`Self::transform_point3()`] instead.
    #[inline]
    #[must_use]
    pub fn transform_vector3(&self, rhs: GVec3<T>) -> GVec3<T> {
        #[allow(clippy::useless_conversion)]
        ((self.matrix3.x_axis * rhs.x)
            + (self.matrix3.y_axis * rhs.y)
            + (self.matrix3.z_axis * rhs.z))
            .into()
    }

    /// Return the inverse of this transform.
    ///
    /// Note that if the transform is not invertible the result will be invalid.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let matrix3 = self.matrix3.inverse();
        let translation = -(matrix3 * self.translation);

        Self {
            matrix3,
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
        self.matrix3.abs_diff_eq(rhs.matrix3, max_abs_diff)
            & self.translation.abs_diff_eq(rhs.translation, max_abs_diff)
    }
}

impl<T: Float> GAffine3<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> T::Bool {
        self.matrix3.is_finite() & self.translation.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> T::Bool {
        self.matrix3.is_nan() | self.translation.is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> GAffine3<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar affine into a SIMD affine, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: GAffine3<T::Element>) -> Self {
        Self {
            matrix3: GMat3::broadcast(value.matrix3),
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
    pub fn extract(&self, i: usize) -> GAffine3<T::Element> {
        GAffine3 {
            matrix3: self.matrix3.extract(i),
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
    pub unsafe fn extract_unchecked(&self, i: usize) -> GAffine3<T::Element> {
        unsafe {
            GAffine3 {
                matrix3: self.matrix3.extract_unchecked(i),
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
    pub fn replace(&mut self, i: usize, value: GAffine3<T::Element>) {
        self.matrix3.replace(i, value.matrix3);
        self.translation.replace(i, value.translation);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: GAffine3<T::Element>) {
        unsafe {
            self.matrix3.replace_unchecked(i, value.matrix3);
            self.translation.replace_unchecked(i, value.translation);
        }
    }
}

/// # Conversion
impl<T: Real> GAffine3<T> {
    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GAffine3<U>
    where
        T: NumCast<U>,
    {
        GAffine3 {
            matrix3: self.matrix3.cast(),
            translation: self.translation.cast(),
        }
    }
}

impl<T: Real> Default for GAffine3<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real> From<GAffine3<T>> for GMat4<T> {
    #[inline]
    fn from(affine: GAffine3<T>) -> Self {
        Self::from_cols(
            affine.matrix3.x_axis.extend(T::ZERO),
            affine.matrix3.y_axis.extend(T::ZERO),
            affine.matrix3.z_axis.extend(T::ZERO),
            affine.translation.extend(T::ONE),
        )
    }
}

impl<T: Real + Mul<Output = T>> Mul for GAffine3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: GAffine3<T>) -> Self {
        Self {
            matrix3: self.matrix3 * rhs.matrix3,
            translation: self.matrix3 * rhs.translation + self.translation,
        }
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GAffine3<T>> for GAffine3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &GAffine3<T>) -> Self {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GAffine3<T>> for &GAffine3<T> {
    type Output = GAffine3<T>;
    #[inline]
    fn mul(self, rhs: GAffine3<T>) -> GAffine3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GAffine3<T>> for &GAffine3<T> {
    type Output = GAffine3<T>;
    #[inline]
    fn mul(self, rhs: &GAffine3<T>) -> GAffine3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign for GAffine3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GAffine3<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + MulAssign> MulAssign<&GAffine3<T>> for GAffine3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GAffine3<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat4<T>> for GAffine3<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: GMat4<T>) -> GMat4<T> {
        GMat4::from(self) * rhs
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat4<T>> for GAffine3<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &GMat4<T>) -> GMat4<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat4<T>> for &GAffine3<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: GMat4<T>) -> GMat4<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat4<T>> for &GAffine3<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &GMat4<T>) -> GMat4<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GAffine3<T>> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: GAffine3<T>) -> GMat4<T> {
        self * GMat4::from(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GAffine3<T>> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &GAffine3<T>) -> GMat4<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GAffine3<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: GAffine3<T>) -> GMat4<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GAffine3<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &GAffine3<T>) -> GMat4<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<GAffine3<T>> for GMat4<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GAffine3<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&GAffine3<T>> for GMat4<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GAffine3<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real> Product<GAffine3<T>> for GAffine3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl<'a, T: Real> Product<&'a GAffine3<T>> for GAffine3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| a * b)
    }
}

impl<T: Real> Deref for GAffine3<T> {
    type Target = crate::deref::Cols4<GVec3<T>>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl<T: Real> DerefMut for GAffine3<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self as *mut Self::Target) }
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for GAffine3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(p) = f.precision() {
            write!(
                f,
                "[{:.*}, {:.*}, {:.*}, {:.*}]",
                p,
                self.matrix3.x_axis,
                p,
                self.matrix3.y_axis,
                p,
                self.matrix3.z_axis,
                p,
                self.translation
            )
        } else {
            write!(
                f,
                "[{}, {}, {}, {}]",
                self.matrix3.x_axis, self.matrix3.y_axis, self.matrix3.z_axis, self.translation
            )
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for GAffine3<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(GAffine3))
            .field("matrix3", &self.matrix3)
            .field("translation", &self.translation)
            .finish()
    }
}
