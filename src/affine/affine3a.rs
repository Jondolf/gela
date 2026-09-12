use crate::affine::GAffine3;
use crate::matrix::{GMat3, GMat4, Mat3A};
use crate::rotation::GRot3;
use crate::vector::{GVec3, Vec3A};

use core::{iter::Product, ops::*};

use gnum::num::{NumCast, Real};

/// A 3D affine transform with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Affine3`](crate::affine::Affine3),
/// at the cost of a larger size.
///
/// It can represent translation, rotation, scaling and shearing.
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
pub struct Affine3A {
    /// The 3x3 matrix containing the rotation and scale of the affine transformation.
    pub matrix3: Mat3A,
    /// The translation vector of the affine transformation.
    pub translation: Vec3A,
}

/// # Constants
impl Affine3A {
    /// The degenerate zero transform.
    ///
    /// This transforms any finite vector and point to zero.
    /// The zero transform is non-invertible.
    pub const ZERO: Self = Self::from_cols(Vec3A::ZERO, Vec3A::ZERO, Vec3A::ZERO, Vec3A::ZERO);

    /// The identity transform.
    ///
    /// Multiplying a vector with this returns the same vector.
    pub const IDENTITY: Self = Self::from_cols(Vec3A::X, Vec3A::Y, Vec3A::Z, Vec3A::ZERO);

    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(Vec3A::NAN, Vec3A::NAN, Vec3A::NAN, Vec3A::NAN);
}

/// # Construction
impl Affine3A {
    /// Creates an affine transform from four column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: Vec3A, y_axis: Vec3A, z_axis: Vec3A, w_axis: Vec3A) -> Self {
        Self {
            matrix3: Mat3A::from_cols(x_axis, y_axis, z_axis),
            translation: w_axis,
        }
    }

    /// Creates an affine transform from a `[f32; 12]` array stored in column major order.
    #[inline]
    #[must_use]
    pub fn from_cols_array(m: &[f32; 12]) -> Self {
        Self {
            matrix3: Mat3A::from_cols_array(&[
                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
            ]),
            translation: Vec3A::from_array([m[9], m[10], m[11]]),
        }
    }

    /// Creates a `[f32; 12]` array storing data in column major order.
    #[inline]
    #[must_use]
    pub fn to_cols_array(&self) -> [f32; 12] {
        let x = &self.matrix3.x_axis;
        let y = &self.matrix3.y_axis;
        let z = &self.matrix3.z_axis;
        let w = &self.translation;
        [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z, w.x, w.y, w.z]
    }

    /// Creates an affine transform from a `[[f32; 3]; 4]` 2D array stored in column major order.
    ///
    /// If your data is in row major order you will need to `transpose` the returned matrix.
    #[inline]
    #[must_use]
    pub fn from_cols_array_2d(m: &[[f32; 3]; 4]) -> Self {
        Self {
            matrix3: Mat3A::from_cols(m[0].into(), m[1].into(), m[2].into()),
            translation: m[3].into(),
        }
    }

    /// Creates a `[[f32; 3]; 4]` 2D array storing data in column major order.
    ///
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array_2d(&self) -> [[f32; 3]; 4] {
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
    pub fn from_cols_slice(slice: &[f32]) -> Self {
        Self {
            matrix3: Mat3A::from_cols_slice(&slice[0..9]),
            translation: Vec3A::from_slice(&slice[9..12]),
        }
    }

    /// Writes the columns of `self` to the first 12 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 12 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [f32]) {
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
    pub fn select(boolean: bool, if_true: Self, if_false: Self) -> Self {
        if boolean { if_true } else { if_false }
    }

    /// Creates an affine transform that changes scale.
    ///
    /// Note that if any scale is zero the transform will be non-invertible.
    #[inline]
    #[must_use]
    pub fn from_scale(scale: GVec3<f32>) -> Self {
        GAffine3::from_scale(scale).into()
    }

    /// Creates an affine transform from the given `rotation`.
    #[inline]
    #[must_use]
    #[doc(alias = "from_quat")]
    pub fn from_rotation(rotation: GRot3<f32>) -> Self {
        GAffine3::from_rotation(rotation).into()
    }

    /// Creates an affine transform containing a 3D rotation around a normalized
    /// rotation `axis` of `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: GVec3<f32>, angle: f32) -> Self {
        GAffine3::from_axis_angle(axis, angle).into()
    }

    /// Creates an affine transform containing a 3D rotation around the x axis of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        GAffine3::from_rotation_x(angle).into()
    }

    /// Creates an affine transform containing a 3D rotation around the y axis of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        GAffine3::from_rotation_y(angle).into()
    }

    /// Creates an affine transform containing a 3D rotation around the z axis of
    /// `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        GAffine3::from_rotation_z(angle).into()
    }

    /// Creates an affine transformation from the given 3D `translation`.
    #[inline]
    #[must_use]
    pub fn from_translation(translation: GVec3<f32>) -> Self {
        GAffine3::from_translation(translation).into()
    }

    /// Creates an affine transform from a 3x3 matrix (expressing scale, shear, and rotation).
    #[inline]
    #[must_use]
    pub fn from_mat3(mat3: GMat3<f32>) -> Self {
        GAffine3::from_mat3(mat3).into()
    }

    /// Creates an affine transform from a 3x3 matrix (expressing scale, shear, and rotation)
    /// and a translation vector.
    ///
    /// Equivalent to `Affine3A::from_translation(translation) * Affine3A::from_mat3(mat3)`
    #[inline]
    #[must_use]
    pub fn from_mat3_translation(mat3: GMat3<f32>, translation: GVec3<f32>) -> Self {
        GAffine3::from_mat3_translation(mat3, translation).into()
    }

    /// Creates an affine transform from the given 3D `scale`, `rotation`, and `translation`.
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: GVec3<f32>,
        rotation: GRot3<f32>,
        translation: GVec3<f32>,
    ) -> Self {
        GAffine3::from_scale_rotation_translation(scale, rotation, translation).into()
    }

    /// Creates an affine transform from the given 3D `rotation` and `translation`.
    #[inline]
    #[must_use]
    pub fn from_rotation_translation(rotation: GRot3<f32>, translation: GVec3<f32>) -> Self {
        GAffine3::from_rotation_translation(rotation, translation).into()
    }

    /// Creates an affine transform from a 4x4 matrix.
    ///
    /// The given matrix must be an affine transform and not contain any perspective transform.
    #[inline]
    #[must_use]
    pub fn from_mat4(m: GMat4<f32>) -> Self {
        GAffine3::from_mat4(m).into()
    }

    /// Extracts `scale`, `rotation` and `translation` from `self`.
    ///
    /// The transform is expected to be non-degenerate and without shearing, or the output
    /// will be invalid.
    #[inline]
    #[must_use]
    pub fn to_scale_rotation_translation(&self) -> (GVec3<f32>, GRot3<f32>, GVec3<f32>) {
        self.to_affine3().to_scale_rotation_translation()
    }

    /// Creates a left-handed view transform using a camera position, an up direction, and a facing direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(eye: GVec3<f32>, dir: GVec3<f32>, up: GVec3<f32>) -> Self {
        GAffine3::look_to_lh(eye, dir, up).into()
    }

    /// Creates a right-handed view transform using a camera position, an up direction, and a facing direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(eye: GVec3<f32>, dir: GVec3<f32>, up: GVec3<f32>) -> Self {
        GAffine3::look_to_rh(eye, dir, up).into()
    }

    /// Creates a left-handed view transform using a camera position, an up direction, and a focal
    /// point. For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: GVec3<f32>, center: GVec3<f32>, up: GVec3<f32>) -> Self {
        GAffine3::look_at_lh(eye, center, up).into()
    }

    /// Creates a right-handed view transform using a camera position, an up direction, and a focal
    /// point. For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_at_rh(eye: GVec3<f32>, center: GVec3<f32>, up: GVec3<f32>) -> Self {
        GAffine3::look_at_rh(eye, center, up).into()
    }
}

/// # Operations
impl Affine3A {
    /// Transforms the given 3D points, applying shear, scale, rotation and translation.
    #[inline]
    pub fn transform_point3(&self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.to_affine3().transform_point3(rhs)
    }

    /// Transforms the given SIMD-aligned 3D point, applying shear, scale, rotation and translation.
    #[inline]
    #[must_use]
    pub fn transform_point3a(&self, rhs: Vec3A) -> Vec3A {
        (self.matrix3.x_axis * rhs.x)
            + (self.matrix3.y_axis * rhs.y)
            + (self.matrix3.z_axis * rhs.z)
            + self.translation
    }

    /// Transforms the given 3D vector, applying shear, scale and rotation (but NOT translation).
    ///
    /// To also apply translation, use [`Self::transform_point3()`] instead.
    #[inline]
    #[must_use]
    pub fn transform_vector3(&self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.to_affine3().transform_vector3(rhs)
    }

    /// Transforms the given SIMD-aligned 3D vector, applying shear, scale and rotation
    /// (but NOT translation).
    ///
    /// To also apply translation, use [`Self::transform_point3a()`] instead.
    #[inline]
    #[must_use]
    pub fn transform_vector3a(&self, rhs: Vec3A) -> Vec3A {
        (self.matrix3.x_axis * rhs.x)
            + (self.matrix3.y_axis * rhs.y)
            + (self.matrix3.z_axis * rhs.z)
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
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: f32) -> bool {
        self.matrix3.abs_diff_eq(rhs.matrix3, max_abs_diff)
            & self.translation.abs_diff_eq(rhs.translation, max_abs_diff)
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.matrix3.is_finite() & self.translation.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.matrix3.is_nan() | self.translation.is_nan()
    }
}

/// # Conversions
impl Affine3A {
    /// Creates an [`Affine3A`] from a [`GAffine3<f32>`].
    #[inline(always)]
    #[must_use]
    pub const fn from_affine3(a: GAffine3<f32>) -> Self {
        Self {
            matrix3: Mat3A::from_mat3(a.matrix3),
            translation: Vec3A::from_vec3(a.translation),
        }
    }

    /// Converts `self` to a [`GAffine3<f32>`].
    #[inline(always)]
    #[must_use]
    pub fn to_affine3(self) -> GAffine3<f32> {
        GAffine3 {
            matrix3: self.matrix3.to_mat3(),
            translation: self.translation.to_vec3(),
        }
    }

    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GAffine3<U>
    where
        f32: NumCast<U>,
    {
        self.to_affine3().cast()
    }
}

impl Default for Affine3A {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Affine3A> for GMat4<f32> {
    #[inline]
    fn from(affine: Affine3A) -> Self {
        GMat4::from(affine.to_affine3())
    }
}

impl From<GAffine3<f32>> for Affine3A {
    #[inline(always)]
    fn from(a: GAffine3<f32>) -> Self {
        Self::from_affine3(a)
    }
}

impl From<Affine3A> for GAffine3<f32> {
    #[inline(always)]
    fn from(a: Affine3A) -> Self {
        a.to_affine3()
    }
}

impl Mul for Affine3A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            matrix3: self.matrix3 * rhs.matrix3,
            translation: self.matrix3.mul_vec3a(rhs.translation) + self.translation,
        }
    }
}

impl Mul<&Affine3A> for Affine3A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Affine3A) -> Self {
        self.mul(*rhs)
    }
}

impl MulAssign for Affine3A {
    #[inline]
    fn mul_assign(&mut self, rhs: Affine3A) {
        *self = self.mul(rhs);
    }
}

impl Mul<GMat4<f32>> for Affine3A {
    type Output = GMat4<f32>;
    #[inline]
    fn mul(self, rhs: GMat4<f32>) -> GMat4<f32> {
        GMat4::from(self) * rhs
    }
}

impl Mul<Affine3A> for GMat4<f32> {
    type Output = GMat4<f32>;
    #[inline]
    fn mul(self, rhs: Affine3A) -> GMat4<f32> {
        self * GMat4::from(rhs)
    }
}

impl Product<Affine3A> for Affine3A {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl Deref for Affine3A {
    type Target = crate::deref::Cols4<Vec3A>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl DerefMut for Affine3A {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self as *mut Self::Target) }
    }
}

impl core::fmt::Display for Affine3A {
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

impl core::fmt::Debug for Affine3A {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct("Affine3A")
            .field("matrix3", &self.matrix3)
            .field("translation", &self.translation)
            .finish()
    }
}
