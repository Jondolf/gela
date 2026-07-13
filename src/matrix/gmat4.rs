use crate::matrix::GMat3;
use crate::rotation::{EulerRot, FromEuler, GQuat, ToEuler};
use crate::vector::{GVec3, GVec4, Vec4Swizzles};

use core::{
    iter::{Product, Sum},
    ops::*,
};

use gnum::{
    num::{Float, Real, ScalarReal},
    simd::Select,
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// Creates a 4x4 matrix from four column vectors.
#[inline(always)]
#[must_use]
pub const fn gmat4<T: Real>(
    x_axis: GVec4<T>,
    y_axis: GVec4<T>,
    z_axis: GVec4<T>,
    w_axis: GVec4<T>,
) -> GMat4<T> {
    GMat4::from_cols(x_axis, y_axis, z_axis, w_axis)
}

/// A 3x3 column major matrix.
///
/// This 4x4 matrix type features convenience methods for creating and using affine transforms and
/// perspective projections. If you are primarily dealing with 3D affine transformations
/// considering using [`GAffine3`](crate::affine::GAffine3) which is faster than a 4x4 matrix
/// for some affine operations.
///
/// Affine transformations including 3D translation, rotation and scale can be created
/// using methods such as [`Self::from_translation()`], [`Self::from_quat()`],
/// [`Self::from_scale()`] and [`Self::from_scale_rotation_translation()`].
///
/// Orthographic projections can be created using the methods [`Self::orthographic_lh()`] for
/// left-handed coordinate systems and [`Self::orthographic_rh()`] for right-handed
/// systems. The resulting matrix is also an affine transformation.
///
/// The [`Self::transform_point3()`] and [`Self::transform_vector3()`] convenience methods
/// are provided for performing affine transformations on 3D vectors and points. These
/// multiply 3D inputs as 4D vectors with an implicit `w` value of `1` for points and `0`
/// for vectors respectively. These methods assume that `Self` contains a valid affine
/// transform.
///
/// Perspective projections can be created using methods such as
/// [`Self::perspective_lh()`], [`Self::perspective_infinite_lh()`] and
/// [`Self::perspective_infinite_reverse_lh()`] for left-handed co-ordinate systems and
/// [`Self::perspective_rh()`], [`Self::perspective_infinite_rh()`] and
/// [`Self::perspective_infinite_reverse_rh()`] for right-handed co-ordinate systems.
///
/// The resulting perspective project can be use to transform 3D vectors as points with
/// perspective correction using the [`Self::project_point3()`] convenience method.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[cfg_attr(
    feature = "zerocopy",
    derive(FromBytes, Immutable, IntoBytes, KnownLayout)
)]
#[repr(align(16))]
#[repr(C)]
pub struct GMat4<T: Real> {
    /// The first column of the matrix.
    pub x_axis: GVec4<T>,
    /// The second column of the matrix.
    pub y_axis: GVec4<T>,
    /// The third column of the matrix.
    pub z_axis: GVec4<T>,
    /// The fourth column of the matrix.
    pub w_axis: GVec4<T>,
}

/// # Constants
impl<T: Real> GMat4<T> {
    /// All zeros.
    pub const ZERO: Self = Self::from_cols(GVec4::ZERO, GVec4::ZERO, GVec4::ZERO, GVec4::ZERO);

    /// The 4x4 identity matrix, where all diagonal elements are `1.0` and all off-diagonal elements are `0.0`.
    pub const IDENTITY: Self = Self::from_cols(GVec4::X, GVec4::Y, GVec4::Z, GVec4::W);
}

impl<T: Float> GMat4<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(GVec4::NAN, GVec4::NAN, GVec4::NAN, GVec4::NAN);
}

/// # Constructors
impl<T: Real> GMat4<T> {
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    #[must_use]
    const fn new(
        m00: T,
        m01: T,
        m02: T,
        m03: T,
        m10: T,
        m11: T,
        m12: T,
        m13: T,
        m20: T,
        m21: T,
        m22: T,
        m23: T,
        m30: T,
        m31: T,
        m32: T,
        m33: T,
    ) -> Self {
        Self {
            x_axis: GVec4::new(m00, m01, m02, m03),
            y_axis: GVec4::new(m10, m11, m12, m13),
            z_axis: GVec4::new(m20, m21, m22, m23),
            w_axis: GVec4::new(m30, m31, m32, m33),
        }
    }

    /// Creates a 4x4 matrix from four column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(
        x_axis: GVec4<T>,
        y_axis: GVec4<T>,
        z_axis: GVec4<T>,
        w_axis: GVec4<T>,
    ) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    /// Creates a 4x4 matrix from a `[T; 16]` array stored in column major order.
    /// If your data is stored in row major you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array(m: &[T; 16]) -> Self {
        Self::new(
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15],
        )
    }

    /// Creates a `[T; 16]` array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array(&self) -> [T; 16] {
        [
            self.x_axis.x,
            self.x_axis.y,
            self.x_axis.z,
            self.x_axis.w,
            self.y_axis.x,
            self.y_axis.y,
            self.y_axis.z,
            self.y_axis.w,
            self.z_axis.x,
            self.z_axis.y,
            self.z_axis.z,
            self.z_axis.w,
            self.w_axis.x,
            self.w_axis.y,
            self.w_axis.z,
            self.w_axis.w,
        ]
    }

    /// Creates a 4x4 matrix from a `[[T; 4]; 4]` 2D array stored in column major order.
    /// If your data is in row major order you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[T; 4]; 4]) -> Self {
        Self::from_cols(
            GVec4::from_array(m[0]),
            GVec4::from_array(m[1]),
            GVec4::from_array(m[2]),
            GVec4::from_array(m[3]),
        )
    }

    /// Creates a `[[T; 4]; 4]` 2D array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[T; 4]; 4] {
        [
            self.x_axis.to_array(),
            self.y_axis.to_array(),
            self.z_axis.to_array(),
            self.w_axis.to_array(),
        ]
    }

    /// Creates a 4x4 matrix with its diagonal set to `diagonal` and all other entries set to 0.
    #[doc(alias = "scale")]
    #[inline]
    #[must_use]
    pub const fn from_diagonal(diagonal: GVec4<T>) -> Self {
        Self::new(
            diagonal.x,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            diagonal.y,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            diagonal.z,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            diagonal.w,
        )
    }

    /// Creates a matrix from the elements in `if_true` and `if_false`, selecting which to use
    /// based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select<B: Select<T>>(boolean: B, if_true: Self, if_false: Self) -> Self {
        Self {
            x_axis: GVec4::select(boolean, if_true.x_axis, if_false.x_axis),
            y_axis: GVec4::select(boolean, if_true.y_axis, if_false.y_axis),
            z_axis: GVec4::select(boolean, if_true.z_axis, if_false.z_axis),
            w_axis: GVec4::select(boolean, if_true.w_axis, if_false.w_axis),
        }
    }

    #[inline]
    #[must_use]
    fn quat_to_axes(rotation: GQuat<T>) -> (GVec4<T>, GVec4<T>, GVec4<T>) {
        let (x, y, z, w) = rotation.into();
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        let x_axis = GVec4::new(T::ONE - (yy + zz), xy + wz, xz - wy, T::ZERO);
        let y_axis = GVec4::new(xy - wz, T::ONE - (xx + zz), yz + wx, T::ZERO);
        let z_axis = GVec4::new(xz + wy, yz - wx, T::ONE - (xx + yy), T::ZERO);
        (x_axis, y_axis, z_axis)
    }

    /// Creates an affine transformation matrix from the given 3D `scale`, `rotation` and
    /// `translation`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: GVec3<T>,
        rotation: GQuat<T>,
        translation: GVec3<T>,
    ) -> Self {
        let (x_axis, y_axis, z_axis) = Self::quat_to_axes(rotation);
        Self::from_cols(
            x_axis.mul(scale.x),
            y_axis.mul(scale.y),
            z_axis.mul(scale.z),
            GVec4::from((translation, T::ONE)),
        )
    }

    /// Creates an affine transformation matrix from the given 3D `translation`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_translation(rotation: GQuat<T>, translation: GVec3<T>) -> Self {
        let (x_axis, y_axis, z_axis) = Self::quat_to_axes(rotation);
        Self::from_cols(x_axis, y_axis, z_axis, GVec4::from((translation, T::ONE)))
    }

    /// Extracts `scale`, `rotation` and `translation` from `self`. The input matrix is
    /// expected to be a 3D affine transformation matrix otherwise the output will be invalid.
    #[inline]
    #[must_use]
    pub fn to_scale_rotation_translation(&self) -> (GVec3<T>, GQuat<T>, GVec3<T>) {
        let det = self.determinant();

        let scale = GVec3::new(
            self.x_axis.length() * det.signum(),
            self.y_axis.length(),
            self.z_axis.length(),
        );

        let inv_scale = scale.recip();

        let rotation = GQuat::from_rotation_axes(
            self.x_axis.mul(inv_scale.x).xyz(),
            self.y_axis.mul(inv_scale.y).xyz(),
            self.z_axis.mul(inv_scale.z).xyz(),
        );

        let translation = self.w_axis.xyz();

        (scale, rotation, translation)
    }

    /// Creates an affine transformation matrix from the given `rotation` quaternion.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_quat(rotation: GQuat<T>) -> Self {
        let (x_axis, y_axis, z_axis) = Self::quat_to_axes(rotation);
        Self::from_cols(x_axis, y_axis, z_axis, GVec4::W)
    }

    /// Creates an affine transformation matrix from the given 3x3 linear transformation
    /// matrix.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_mat3(m: GMat3<T>) -> Self {
        Self::from_cols(
            GVec4::from((m.x_axis, T::ZERO)),
            GVec4::from((m.y_axis, T::ZERO)),
            GVec4::from((m.z_axis, T::ZERO)),
            GVec4::W,
        )
    }

    /// Creates an affine transformation matrics from a 3x3 matrix (expressing scale, shear and
    /// rotation) and a translation vector.
    ///
    /// Equivalent to `Mat4::from_translation(translation) * Mat4::from_mat3(mat3)`
    #[inline]
    #[must_use]
    pub fn from_mat3_translation(mat3: GMat3<T>, translation: GVec3<T>) -> Self {
        Self::from_cols(
            GVec4::from((mat3.x_axis, T::ZERO)),
            GVec4::from((mat3.y_axis, T::ZERO)),
            GVec4::from((mat3.z_axis, T::ZERO)),
            GVec4::from((translation, T::ONE)),
        )
    }

    /// Creates an affine transformation matrix from the given 3D `translation`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_translation(translation: GVec3<T>) -> Self {
        Self::from_cols(
            GVec4::X,
            GVec4::Y,
            GVec4::Z,
            GVec4::new(translation.x, translation.y, translation.z, T::ONE),
        )
    }

    /// Creates an affine transformation matrix containing a 3D rotation around a normalized
    /// rotation `axis` of `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: GVec3<T>, angle: T) -> Self {
        let (sin, cos) = angle.sin_cos_stable();
        let axis_sin = axis.mul(sin);
        let axis_sq = axis.mul(axis);
        let omc = T::ONE - cos;
        let xyomc = axis.x * axis.y * omc;
        let xzomc = axis.x * axis.z * omc;
        let yzomc = axis.y * axis.z * omc;
        Self::from_cols(
            GVec4::new(
                axis_sq.x * omc + cos,
                xyomc + axis_sin.z,
                xzomc - axis_sin.y,
                T::ZERO,
            ),
            GVec4::new(
                xyomc - axis_sin.z,
                axis_sq.y * omc + cos,
                yzomc + axis_sin.x,
                T::ZERO,
            ),
            GVec4::new(
                xzomc + axis_sin.y,
                yzomc - axis_sin.x,
                axis_sq.z * omc + cos,
                T::ZERO,
            ),
            GVec4::W,
        )
    }
}

impl<T: ScalarReal> GMat4<T> {
    /// Creates a 3D rotation matrix from the given euler rotation sequence and the angles (in radians).
    #[inline]
    #[must_use]
    pub fn from_euler(order: EulerRot, a: T, b: T, c: T) -> Self {
        Self::from_euler_angles(order, a, b, c)
    }

    /// Extract Euler angles with the given Euler rotation order.
    ///
    /// Note if the input matrix contains scales, shears, or other non-rotation transformations then
    /// the resulting Euler angles will be ill-defined.
    #[inline]
    #[must_use]
    pub fn to_euler(&self, order: EulerRot) -> (T, T, T) {
        self.to_euler_angles(order)
    }
}

impl<T: Real> GMat4<T> {
    /// Creates an affine transformation matrix containing a 3D rotation around the x axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: T) -> Self {
        let (sina, cosa) = angle.sin_cos_stable();
        Self::from_cols(
            GVec4::X,
            GVec4::new(T::ZERO, cosa, sina, T::ZERO),
            GVec4::new(T::ZERO, -sina, cosa, T::ZERO),
            GVec4::W,
        )
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the y axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: T) -> Self {
        let (sina, cosa) = angle.sin_cos_stable();
        Self::from_cols(
            GVec4::new(cosa, T::ZERO, -sina, T::ZERO),
            GVec4::Y,
            GVec4::new(sina, T::ZERO, cosa, T::ZERO),
            GVec4::W,
        )
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the z axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: T) -> Self {
        let (sina, cosa) = angle.sin_cos_stable();
        Self::from_cols(
            GVec4::new(cosa, sina, T::ZERO, T::ZERO),
            GVec4::new(-sina, cosa, T::ZERO, T::ZERO),
            GVec4::Z,
            GVec4::W,
        )
    }

    /// Creates an affine transformation matrix containing the given 3D non-uniform `scale`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_scale(scale: GVec3<T>) -> Self {
        Self::from_cols(
            GVec4::new(scale.x, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, scale.y, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, scale.z, T::ZERO),
            GVec4::W,
        )
    }

    /// Creates a 4x4 matrix from the first 16 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 16 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice(slice: &[T]) -> Self {
        Self::new(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            slice[8], slice[9], slice[10], slice[11], slice[12], slice[13], slice[14], slice[15],
        )
    }

    /// Writes the columns of `self` to the first 16 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 16 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [T]) {
        slice[0] = self.x_axis.x;
        slice[1] = self.x_axis.y;
        slice[2] = self.x_axis.z;
        slice[3] = self.x_axis.w;
        slice[4] = self.y_axis.x;
        slice[5] = self.y_axis.y;
        slice[6] = self.y_axis.z;
        slice[7] = self.y_axis.w;
        slice[8] = self.z_axis.x;
        slice[9] = self.z_axis.y;
        slice[10] = self.z_axis.z;
        slice[11] = self.z_axis.w;
        slice[12] = self.w_axis.x;
        slice[13] = self.w_axis.y;
        slice[14] = self.w_axis.z;
        slice[15] = self.w_axis.w;
    }
}

/// # Operations
impl<T: Real> GMat4<T> {
    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> GVec4<T> {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            3 => self.w_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns a mutable reference to the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    pub fn col_mut(&mut self, index: usize) -> &mut GVec4<T> {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            2 => &mut self.z_axis,
            3 => &mut self.w_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the matrix row for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    #[must_use]
    pub fn row(&self, index: usize) -> GVec4<T> {
        match index {
            0 => GVec4::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
            1 => GVec4::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
            2 => GVec4::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
            3 => GVec4::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            x_axis: GVec4::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
            y_axis: GVec4::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
            z_axis: GVec4::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
            w_axis: GVec4::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
        }
    }

    /// Returns the diagonal of `self`.
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> GVec4<T> {
        GVec4::new(self.x_axis.x, self.y_axis.y, self.z_axis.z, self.w_axis.w)
    }

    /// Returns the determinant of `self`.
    #[must_use]
    pub fn determinant(&self) -> T {
        let (m00, m01, m02, m03) = self.x_axis.into();
        let (m10, m11, m12, m13) = self.y_axis.into();
        let (m20, m21, m22, m23) = self.z_axis.into();
        let (m30, m31, m32, m33) = self.w_axis.into();

        let a2323 = m22 * m33 - m23 * m32;
        let a1323 = m21 * m33 - m23 * m31;
        let a1223 = m21 * m32 - m22 * m31;
        let a0323 = m20 * m33 - m23 * m30;
        let a0223 = m20 * m32 - m22 * m30;
        let a0123 = m20 * m31 - m21 * m30;

        m00 * (m11 * a2323 - m12 * a1323 + m13 * a1223)
            - m01 * (m10 * a2323 - m12 * a0323 + m13 * a0223)
            + m02 * (m10 * a1323 - m11 * a0323 + m13 * a0123)
            - m03 * (m10 * a1223 - m11 * a0223 + m12 * a0123)
    }

    /// If `CHECKED` is true then if the determinant is zero this function will return a tuple
    /// containing a zero matrix and false. If the determinant is non zero a tuple containing the
    /// inverted matrix and true is returned.
    ///
    /// If `CHECKED` is false then the determinant is not checked and if it is zero the resulting
    /// inverted matrix will be invalid. Will panic if the determinant of `self` is zero when
    /// `glam_assert` is enabled.
    ///
    /// A tuple containing the inverted matrix and a bool is used instead of an option here as
    /// regular Rust enums put the discriminant first which can result in a lot of padding if the
    /// matrix is aligned.
    #[inline(always)]
    #[must_use]
    fn inverse_checked<const CHECKED: bool>(&self) -> (Self, T::Bool) {
        let (m00, m01, m02, m03) = self.x_axis.into();
        let (m10, m11, m12, m13) = self.y_axis.into();
        let (m20, m21, m22, m23) = self.z_axis.into();
        let (m30, m31, m32, m33) = self.w_axis.into();

        let coef00 = m22 * m33 - m32 * m23;
        let coef02 = m12 * m33 - m32 * m13;
        let coef03 = m12 * m23 - m22 * m13;

        let coef04 = m21 * m33 - m31 * m23;
        let coef06 = m11 * m33 - m31 * m13;
        let coef07 = m11 * m23 - m21 * m13;

        let coef08 = m21 * m32 - m31 * m22;
        let coef10 = m11 * m32 - m31 * m12;
        let coef11 = m11 * m22 - m21 * m12;

        let coef12 = m20 * m33 - m30 * m23;
        let coef14 = m10 * m33 - m30 * m13;
        let coef15 = m10 * m23 - m20 * m13;

        let coef16 = m20 * m32 - m30 * m22;
        let coef18 = m10 * m32 - m30 * m12;
        let coef19 = m10 * m22 - m20 * m12;

        let coef20 = m20 * m31 - m30 * m21;
        let coef22 = m10 * m31 - m30 * m11;
        let coef23 = m10 * m21 - m20 * m11;

        let fac0 = GVec4::new(coef00, coef00, coef02, coef03);
        let fac1 = GVec4::new(coef04, coef04, coef06, coef07);
        let fac2 = GVec4::new(coef08, coef08, coef10, coef11);
        let fac3 = GVec4::new(coef12, coef12, coef14, coef15);
        let fac4 = GVec4::new(coef16, coef16, coef18, coef19);
        let fac5 = GVec4::new(coef20, coef20, coef22, coef23);

        let vec0 = GVec4::new(m10, m00, m00, m00);
        let vec1 = GVec4::new(m11, m01, m01, m01);
        let vec2 = GVec4::new(m12, m02, m02, m02);
        let vec3 = GVec4::new(m13, m03, m03, m03);

        let inv0 = vec1.mul(fac0).sub(vec2.mul(fac1)).add(vec3.mul(fac2));
        let inv1 = vec0.mul(fac0).sub(vec2.mul(fac3)).add(vec3.mul(fac4));
        let inv2 = vec0.mul(fac1).sub(vec1.mul(fac3)).add(vec3.mul(fac5));
        let inv3 = vec0.mul(fac2).sub(vec1.mul(fac4)).add(vec2.mul(fac5));

        let sign_a = GVec4::new(T::ONE, T::NEG_ONE, T::ONE, T::NEG_ONE);
        let sign_b = GVec4::new(T::NEG_ONE, T::ONE, T::NEG_ONE, T::ONE);
        let inverse = Self::from_cols(
            inv0.mul(sign_a),
            inv1.mul(sign_b),
            inv2.mul(sign_a),
            inv3.mul(sign_b),
        );

        let col0 = GVec4::new(
            inverse.x_axis.x,
            inverse.y_axis.x,
            inverse.z_axis.x,
            inverse.w_axis.x,
        );

        let dot0 = self.x_axis.mul(col0);
        let dot1 = dot0.x + dot0.y + dot0.z + dot0.w;

        let inv_det = dot1.recip();
        let inverted = inverse.mul(inv_det);
        let invertible = dot1.num_ne(T::ZERO);

        (Self::select(invertible, inverted, Self::ZERO), invertible)
    }

    /// Returns the inverse of `self`.
    ///
    /// If the matrix is not invertible the returned matrix will be invalid.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        self.inverse_checked::<false>().0
    }

    /// Returns the inverse of `self` or `Mat2::ZERO` if the matrix is not invertible.
    #[inline]
    #[must_use]
    pub fn inverse_or_zero(&self) -> Self {
        self.inverse_checked::<true>().0
    }

    /// Creates a left-handed view matrix using a camera position, a facing direction, and an up
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(eye: GVec3<T>, dir: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_rh(eye, -dir, up)
    }

    /// Creates a right-handed view matrix using a camera position, a facing direction, and an up
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(eye: GVec3<T>, dir: GVec3<T>, up: GVec3<T>) -> Self {
        let f = dir;
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self::from_cols(
            GVec4::new(s.x, u.x, -f.x, T::ZERO),
            GVec4::new(s.y, u.y, -f.y, T::ZERO),
            GVec4::new(s.z, u.z, -f.z, T::ZERO),
            GVec4::new(-eye.dot(s), -eye.dot(u), eye.dot(f), T::ONE),
        )
    }

    /// Creates a left-handed view matrix using a camera position, a focal point, and an up
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: GVec3<T>, center: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_lh(eye, center.sub(eye).normalize(), up)
    }

    /// Creates a right-handed view matrix using a camera position, a focal point, and an up
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    pub fn look_at_rh(eye: GVec3<T>, center: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_rh(eye, center.sub(eye).normalize(), up)
    }

    /// Creates a right-handed perspective projection matrix with [-1,1] depth range.
    ///
    /// This is the same as the OpenGL `glFrustum` function.
    ///
    /// See <https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml>
    #[inline]
    #[must_use]
    pub fn frustum_rh_gl(left: T, right: T, bottom: T, top: T, z_near: T, z_far: T) -> Self {
        let two = T::ONE + T::ONE;
        let inv_width = T::ONE / (right - left);
        let inv_height = T::ONE / (top - bottom);
        let inv_depth = T::ONE / (z_far - z_near);
        let a = (right + left) * inv_width;
        let b = (top + bottom) * inv_height;
        let c = -(z_far + z_near) * inv_depth;
        let d = -(two * z_far * z_near) * inv_depth;
        let two_z_near = two * z_near;
        Self::from_cols(
            GVec4::new(two_z_near * inv_width, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, two_z_near * inv_height, T::ZERO, T::ZERO),
            GVec4::new(a, b, c, T::NEG_ONE),
            GVec4::new(T::ZERO, T::ZERO, d, T::ZERO),
        )
    }

    /// Creates a left-handed perspective projection matrix with `[0,1]` depth range.
    #[inline]
    #[must_use]
    pub fn frustum_lh(left: T, right: T, bottom: T, top: T, z_near: T, z_far: T) -> Self {
        let two = T::ONE + T::ONE;
        let inv_width = T::ONE / (right - left);
        let inv_height = T::ONE / (top - bottom);
        let inv_depth = T::ONE / (z_far - z_near);
        let a = (right + left) * inv_width;
        let b = (top + bottom) * inv_height;
        let c = z_far * inv_depth;
        let d = -(z_far * z_near) * inv_depth;
        let two_z_near = two * z_near;
        Self::from_cols(
            GVec4::new(two_z_near * inv_width, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, two_z_near * inv_height, T::ZERO, T::ZERO),
            GVec4::new(a, b, c, T::ONE),
            GVec4::new(T::ZERO, T::ZERO, d, T::ZERO),
        )
    }

    /// Creates a right-handed perspective projection matrix with `[0,1]` depth range.
    #[inline]
    #[must_use]
    pub fn frustum_rh(left: T, right: T, bottom: T, top: T, z_near: T, z_far: T) -> Self {
        let two = T::ONE + T::ONE;
        let inv_width = T::ONE / (right - left);
        let inv_height = T::ONE / (top - bottom);
        let inv_depth = T::ONE / (z_far - z_near);
        let a = (right + left) * inv_width;
        let b = (top + bottom) * inv_height;
        let c = -z_far * inv_depth;
        let d = -(z_far * z_near) * inv_depth;
        let two_z_near = two * z_near;
        Self::from_cols(
            GVec4::new(two_z_near * inv_width, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, two_z_near * inv_height, T::ZERO, T::ZERO),
            GVec4::new(a, b, c, T::NEG_ONE),
            GVec4::new(T::ZERO, T::ZERO, d, T::ZERO),
        )
    }

    /// Creates a right-handed perspective projection matrix with `[-1,1]` depth range.
    ///
    /// Useful to map the standard right-handed coordinate system into what OpenGL expects.
    ///
    /// This is the same as the OpenGL `gluPerspective` function.
    /// See <https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml>
    #[inline]
    #[must_use]
    pub fn perspective_rh_gl(fov_y_radians: T, aspect_ratio: T, z_near: T, z_far: T) -> Self {
        let two = T::ONE + T::ONE;
        let inv_length = T::ONE / (z_near - z_far);
        let f = T::ONE / (T::HALF * fov_y_radians).tan();
        let a = f / aspect_ratio;
        let b = (z_near + z_far) * inv_length;
        let c = (two * z_near * z_far) * inv_length;
        Self::from_cols(
            GVec4::new(a, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, f, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, b, -T::ONE),
            GVec4::new(T::ZERO, T::ZERO, c, T::ZERO),
        )
    }

    /// Creates a left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Useful to map the standard left-handed coordinate system into what WebGPU/Metal/Direct3D expect.
    #[inline]
    #[must_use]
    pub fn perspective_lh(fov_y_radians: T, aspect_ratio: T, z_near: T, z_far: T) -> Self {
        let (sin_fov, cos_fov) = (T::HALF * fov_y_radians).sin_cos_stable();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        let r = z_far / (z_far - z_near);
        Self::from_cols(
            GVec4::new(w, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, h, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, r, T::ONE),
            GVec4::new(T::ZERO, T::ZERO, -r * z_near, T::ZERO),
        )
    }

    /// Creates a right-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Useful to map the standard right-handed coordinate system into what WebGPU/Metal/Direct3D expect.
    #[inline]
    #[must_use]
    pub fn perspective_rh(fov_y_radians: T, aspect_ratio: T, z_near: T, z_far: T) -> Self {
        let (sin_fov, cos_fov) = (T::HALF * fov_y_radians).sin_cos_stable();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        let r = z_far / (z_near - z_far);
        Self::from_cols(
            GVec4::new(w, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, h, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, r, -T::ONE),
            GVec4::new(T::ZERO, T::ZERO, r * z_near, T::ZERO),
        )
    }

    /// Creates an infinite left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Like `perspective_lh`, but with an infinite value for `z_far`.
    /// The result is that points near `z_near` are mapped to depth `0`, and as they move towards infinity the depth approaches `1`.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_lh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self {
        let (sin_fov, cos_fov) = (T::HALF * fov_y_radians).sin_cos_stable();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        Self::from_cols(
            GVec4::new(w, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, h, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, T::ONE, T::ONE),
            GVec4::new(T::ZERO, T::ZERO, -z_near, T::ZERO),
        )
    }

    /// Creates an infinite reverse left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Similar to `perspective_infinite_lh`, but maps `Z = z_near` to a depth of `1` and `Z = infinity` to a depth of `0`.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_reverse_lh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self {
        let (sin_fov, cos_fov) = (T::HALF * fov_y_radians).sin_cos_stable();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        Self::from_cols(
            GVec4::new(w, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, h, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, T::ZERO, T::ONE),
            GVec4::new(T::ZERO, T::ZERO, z_near, T::ZERO),
        )
    }

    /// Creates an infinite right-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Like `perspective_rh`, but with an infinite value for `z_far`.
    /// The result is that points near `z_near` are mapped to depth `0`, and as they move towards infinity the depth approaches `1`.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_rh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self {
        let f = T::ONE / (T::HALF * fov_y_radians).tan();
        Self::from_cols(
            GVec4::new(f / aspect_ratio, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, f, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, -T::ONE, -T::ONE),
            GVec4::new(T::ZERO, T::ZERO, -z_near, T::ZERO),
        )
    }

    /// Creates an infinite reverse right-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Similar to `perspective_infinite_rh`, but maps `Z = z_near` to a depth of `1` and `Z = infinity` to a depth of `0`.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_reverse_rh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self {
        let f = T::ONE / (T::HALF * fov_y_radians).tan();
        Self::from_cols(
            GVec4::new(f / aspect_ratio, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, f, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, T::ZERO, -T::ONE),
            GVec4::new(T::ZERO, T::ZERO, z_near, T::ZERO),
        )
    }

    /// Creates a right-handed orthographic projection matrix with `[-1,1]` depth
    /// range.  This is the same as the OpenGL `glOrtho` function in OpenGL.
    /// See
    /// <https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml>
    ///
    /// Useful to map a right-handed coordinate system to the normalized device coordinates that OpenGL expects.
    #[inline]
    #[must_use]
    pub fn orthographic_rh_gl(left: T, right: T, bottom: T, top: T, near: T, far: T) -> Self {
        let two = T::ONE + T::ONE;
        let a = two / (right - left);
        let b = two / (top - bottom);
        let c = -two / (far - near);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);

        Self::from_cols(
            GVec4::new(a, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, b, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, c, T::ZERO),
            GVec4::new(tx, ty, tz, T::ONE),
        )
    }

    /// Creates a left-handed orthographic projection matrix with `[0,1]` depth range.
    ///
    /// Useful to map a left-handed coordinate system to the normalized device coordinates that WebGPU/Direct3D/Metal expect.
    #[inline]
    #[must_use]
    pub fn orthographic_lh(left: T, right: T, bottom: T, top: T, near: T, far: T) -> Self {
        let rcp_width = T::ONE / (right - left);
        let rcp_height = T::ONE / (top - bottom);
        let r = T::ONE / (far - near);
        Self::from_cols(
            GVec4::new(rcp_width + rcp_width, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, rcp_height + rcp_height, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, r, T::ZERO),
            GVec4::new(
                -(left + right) * rcp_width,
                -(top + bottom) * rcp_height,
                -r * near,
                T::ONE,
            ),
        )
    }

    /// Creates a right-handed orthographic projection matrix with `[0,1]` depth range.
    ///
    /// Useful to map a right-handed coordinate system to the normalized device coordinates that WebGPU/Direct3D/Metal expect.
    #[inline]
    #[must_use]
    pub fn orthographic_rh(left: T, right: T, bottom: T, top: T, near: T, far: T) -> Self {
        let rcp_width = T::ONE / (right - left);
        let rcp_height = T::ONE / (top - bottom);
        let r = T::ONE / (near - far);
        Self::from_cols(
            GVec4::new(rcp_width + rcp_width, T::ZERO, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, rcp_height + rcp_height, T::ZERO, T::ZERO),
            GVec4::new(T::ZERO, T::ZERO, r, T::ZERO),
            GVec4::new(
                -(left + right) * rcp_width,
                -(top + bottom) * rcp_height,
                r * near,
                T::ONE,
            ),
        )
    }

    /// Transforms the given 3D vector as a point, applying perspective correction.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is `1.0`.
    /// The perspective divide is performed meaning the resulting 3D vector is divided by `w`.
    ///
    /// This method assumes that `self` contains a projective transform.
    #[inline]
    #[must_use]
    pub fn project_point3(&self, rhs: GVec3<T>) -> GVec3<T> {
        let mut res = self.x_axis.mul(rhs.x);
        res = self.y_axis.mul(rhs.y).add(res);
        res = self.z_axis.mul(rhs.z).add(res);
        res = self.w_axis.add(res);
        res = res.div(res.w);
        res.xyz()
    }

    /// Transforms the given 3D vector as a point.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is
    /// `1.0`.
    ///
    /// This method assumes that `self` contains a valid affine transform. It does not perform
    /// a perspective divide, if `self` contains a perspective transform, or if you are unsure,
    /// the [`Self::project_point3()`] method should be used instead.
    #[inline]
    #[must_use]
    pub fn transform_point3(&self, rhs: GVec3<T>) -> GVec3<T> {
        let mut res = self.x_axis.mul(rhs.x);
        res = self.y_axis.mul(rhs.y).add(res);
        res = self.z_axis.mul(rhs.z).add(res);
        res = self.w_axis.add(res);
        res.xyz()
    }

    /// Transforms the give 3D vector as a direction.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is
    /// `0.0`.
    ///
    /// This method assumes that `self` contains a valid affine transform.
    #[inline]
    #[must_use]
    pub fn transform_vector3(&self, rhs: GVec3<T>) -> GVec3<T> {
        let mut res = self.x_axis.mul(rhs.x);
        res = self.y_axis.mul(rhs.y).add(res);
        res = self.z_axis.mul(rhs.z).add(res);
        res.xyz()
    }

    /// Transforms a 4D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec4(&self, rhs: GVec4<T>) -> GVec4<T> {
        let mut res = self.x_axis.mul(rhs.x);
        res = res.add(self.y_axis.mul(rhs.y));
        res = res.add(self.z_axis.mul(rhs.z));
        res = res.add(self.w_axis.mul(rhs.w));
        res
    }

    /// Transforms a 4D vector by the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn mul_transpose_vec4(&self, rhs: GVec4<T>) -> GVec4<T> {
        GVec4::new(
            self.x_axis.dot(rhs),
            self.y_axis.dot(rhs),
            self.z_axis.dot(rhs),
            self.w_axis.dot(rhs),
        )
    }

    /// Multiplies two 4x4 matrices.
    #[inline]
    #[must_use]
    pub fn mul_mat4(&self, rhs: &Self) -> Self {
        self.mul(rhs)
    }

    /// Adds two 4x4 matrices.
    #[inline]
    #[must_use]
    pub fn add_mat4(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }

    /// Subtracts two 4x4 matrices.
    #[inline]
    #[must_use]
    pub fn sub_mat4(&self, rhs: &Self) -> Self {
        self.sub(rhs)
    }

    /// Multiplies a 4x4 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn mul_scalar(&self, rhs: T) -> Self {
        Self::from_cols(
            self.x_axis.mul(rhs),
            self.y_axis.mul(rhs),
            self.z_axis.mul(rhs),
            self.w_axis.mul(rhs),
        )
    }

    /// Multiply `self` by a scaling vector `scale`.
    ///
    /// This is faster than creating a whole diagonal scaling matrix and then multiplying that.
    /// This operation is commutative.
    #[inline]
    #[must_use]
    pub fn mul_diagonal_scale(&self, scale: GVec4<T>) -> Self {
        Self::from_cols(
            self.x_axis * scale.x,
            self.y_axis * scale.y,
            self.z_axis * scale.z,
            self.w_axis * scale.w,
        )
    }

    /// Divides a 4x4 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn div_scalar(&self, rhs: T) -> Self {
        let rhs = GVec4::splat(rhs);
        Self::from_cols(
            self.x_axis.div(rhs),
            self.y_axis.div(rhs),
            self.z_axis.div(rhs),
            self.w_axis.div(rhs),
        )
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs`
    /// is less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two matrices contain similar elements. It works best
    /// when comparing with a known value. The `max_abs_diff` that should be used used
    /// depends on the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: T) -> T::Bool {
        self.x_axis.abs_diff_eq(rhs.x_axis, max_abs_diff)
            & self.y_axis.abs_diff_eq(rhs.y_axis, max_abs_diff)
            & self.z_axis.abs_diff_eq(rhs.z_axis, max_abs_diff)
            & self.w_axis.abs_diff_eq(rhs.w_axis, max_abs_diff)
    }

    /// Takes the absolute value of each element in `self`
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self::from_cols(
            self.x_axis.abs(),
            self.y_axis.abs(),
            self.z_axis.abs(),
            self.w_axis.abs(),
        )
    }
}

impl<T: Float> GMat4<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> T::Bool {
        self.x_axis.is_finite()
            & self.y_axis.is_finite()
            & self.z_axis.is_finite()
            & self.w_axis.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> T::Bool {
        self.x_axis.is_nan() | self.y_axis.is_nan() | self.z_axis.is_nan() | self.w_axis.is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> GMat4<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar matrix into a SIMD matrix, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: GMat4<T::Element>) -> Self {
        Self::from_cols(
            GVec4::broadcast(value.x_axis),
            GVec4::broadcast(value.y_axis),
            GVec4::broadcast(value.z_axis),
            GVec4::broadcast(value.w_axis),
        )
    }

    /// Extracts the i-th lane of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub fn extract(&self, i: usize) -> GMat4<T::Element> {
        GMat4::from_cols(
            self.x_axis.extract(i),
            self.y_axis.extract(i),
            self.z_axis.extract(i),
            self.w_axis.extract(i),
        )
    }

    /// Extracts the i-th lane of `self` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub unsafe fn extract_unchecked(&self, i: usize) -> GMat4<T::Element> {
        unsafe {
            GMat4::from_cols(
                self.x_axis.extract_unchecked(i),
                self.y_axis.extract_unchecked(i),
                self.z_axis.extract_unchecked(i),
                self.w_axis.extract_unchecked(i),
            )
        }
    }

    /// Replaces the i-th lane of `self` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    pub fn replace(&mut self, i: usize, value: GMat4<T::Element>) {
        self.x_axis.replace(i, value.x_axis);
        self.y_axis.replace(i, value.y_axis);
        self.z_axis.replace(i, value.z_axis);
        self.w_axis.replace(i, value.w_axis);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: GMat4<T::Element>) {
        unsafe {
            self.x_axis.replace_unchecked(i, value.x_axis);
            self.y_axis.replace_unchecked(i, value.y_axis);
            self.z_axis.replace_unchecked(i, value.z_axis);
            self.w_axis.replace_unchecked(i, value.w_axis);
        }
    }
}

impl<T: Real> Default for GMat4<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real + Add<Output = T>> Add for GMat4<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: GMat4<T>) -> Self {
        GMat4::from_cols(
            self.x_axis.add(rhs.x_axis),
            self.y_axis.add(rhs.y_axis),
            self.z_axis.add(rhs.z_axis),
            self.w_axis.add(rhs.w_axis),
        )
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat4<T>> for GMat4<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &GMat4<T>) -> Self {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<GMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: GMat4<T>) -> GMat4<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: &GMat4<T>) -> GMat4<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> AddAssign for GMat4<T> {
    #[inline]
    fn add_assign(&mut self, rhs: GMat4<T>) {
        *self = self.add(rhs);
    }
}

impl<T: Real + AddAssign> AddAssign<&GMat4<T>> for GMat4<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &GMat4<T>) {
        self.add_assign(*rhs);
    }
}

impl<T: Real + Sub<Output = T>> Sub for GMat4<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: GMat4<T>) -> Self {
        GMat4::from_cols(
            self.x_axis.sub(rhs.x_axis),
            self.y_axis.sub(rhs.y_axis),
            self.z_axis.sub(rhs.z_axis),
            self.w_axis.sub(rhs.w_axis),
        )
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat4<T>> for GMat4<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &GMat4<T>) -> Self {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<GMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: GMat4<T>) -> GMat4<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: &GMat4<T>) -> GMat4<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> SubAssign for GMat4<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: GMat4<T>) {
        *self = self.sub(rhs);
    }
}

impl<T: Real + SubAssign> SubAssign<&GMat4<T>> for GMat4<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &GMat4<T>) {
        self.sub_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul for GMat4<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: GMat4<T>) -> Self {
        GMat4::from_cols(
            self.x_axis.mul(rhs.x_axis),
            self.y_axis.mul(rhs.y_axis),
            self.z_axis.mul(rhs.z_axis),
            self.w_axis.mul(rhs.w_axis),
        )
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat4<T>> for GMat4<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &GMat4<T>) -> Self {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: GMat4<T>) -> GMat4<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &GMat4<T>) -> GMat4<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign for GMat4<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GMat4<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + MulAssign> MulAssign<&GMat4<T>> for GMat4<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GMat4<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec4<T>> for GMat4<T> {
    type Output = GVec4<T>;
    #[inline]
    fn mul(self, rhs: GVec4<T>) -> GVec4<T> {
        self.mul_vec4(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec4<T>> for GMat4<T> {
    type Output = GVec4<T>;
    #[inline]
    fn mul(self, rhs: &GVec4<T>) -> GVec4<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec4<T>> for &GMat4<T> {
    type Output = GVec4<T>;
    #[inline]
    fn mul(self, rhs: GVec4<T>) -> GVec4<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec4<T>> for &GMat4<T> {
    type Output = GVec4<T>;
    #[inline]
    fn mul(self, rhs: &GVec4<T>) -> GVec4<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: T) -> GMat4<T> {
        self.mul_scalar(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GMat4<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: T) -> GMat4<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GMat4<T> {
        (*self).mul(*rhs)
    }
}

// We cannot implement scalar * matrix generically because of Rust's orphan rules.
macro_rules! impl_scalar_left_mul {
    ($($t:ty),*) => {
        $(
            impl Mul<GMat4<$t>> for $t {
                type Output = GMat4<$t>;
                #[inline]
                fn mul(self, rhs: GMat4<$t>) -> GMat4<$t> {
                    rhs.mul_scalar(self)
                }
            }

            impl Mul<&GMat4<$t>> for $t {
                type Output = GMat4<$t>;
                #[inline]
                fn mul(self, rhs: &GMat4<$t>) -> GMat4<$t> {
                    self.mul(*rhs)
                }
            }

            impl Mul<GMat4<$t>> for &$t {
                type Output = GMat4<$t>;
                #[inline]
                fn mul(self, rhs: GMat4<$t>) -> GMat4<$t> {
                    (*self).mul(rhs)
                }
            }

            impl Mul<&GMat4<$t>> for &$t {
                type Output = GMat4<$t>;
                #[inline]
                fn mul(self, rhs: &GMat4<$t>) -> GMat4<$t> {
                    (*self).mul(*rhs)
                }
            }
        )*
    };
}

// TODO: Implement for SIMD types
impl_scalar_left_mul!(f32, f64);

impl<T: Real + Mul<Output = T>> MulAssign<T> for GMat4<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&T> for GMat4<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Div<Output = T>> Div<T> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn div(self, rhs: T) -> GMat4<T> {
        self.div_scalar(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn div(self, rhs: &T) -> GMat4<T> {
        self.div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn div(self, rhs: T) -> GMat4<T> {
        (*self).div(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn div(self, rhs: &T) -> GMat4<T> {
        (*self).div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> DivAssign<T> for GMat4<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = self.div(rhs);
    }
}

impl<T: Real + Div<Output = T>> DivAssign<&T> for GMat4<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.div_assign(*rhs);
    }
}

impl<T: Real + Neg<Output = T>> Neg for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn neg(self) -> GMat4<T> {
        GMat4::from_cols(
            self.x_axis.neg(),
            self.y_axis.neg(),
            self.z_axis.neg(),
            self.w_axis.neg(),
        )
    }
}

impl<T: Real + Neg<Output = T>> Neg for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn neg(self) -> GMat4<T> {
        (*self).neg()
    }
}

impl<T: Real> Sum<GMat4<T>> for GMat4<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, T: Real> Sum<&'a GMat4<T>> for GMat4<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl<T: Real> Product<GMat4<T>> for GMat4<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl<'a, T: Real> Product<&'a GMat4<T>> for GMat4<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| a * b)
    }
}

impl<T: Real> AsRef<[T; 16]> for GMat4<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 16] {
        unsafe { &*(self as *const GMat4<T> as *const [T; 16]) }
    }
}

impl<T: Real> AsMut<[T; 16]> for GMat4<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 16] {
        unsafe { &mut *(self as *mut GMat4<T> as *mut [T; 16]) }
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for GMat4<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(p) = f.precision() {
            write!(
                f,
                "[{:.*}, {:.*}, {:.*}, {:.*}]",
                p, self.x_axis, p, self.y_axis, p, self.z_axis, p, self.w_axis
            )
        } else {
            write!(
                f,
                "[{}, {}, {}, {}]",
                self.x_axis, self.y_axis, self.z_axis, self.w_axis
            )
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for GMat4<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(GMat4))
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis)
            .field("w_axis", &self.w_axis)
            .finish()
    }
}
