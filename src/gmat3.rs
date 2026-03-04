use crate::{
    EulerRot, FromEuler, GMat2, GMat4, GQuat, GVec2, GVec3, ToEuler, Vec3Swizzles, Vec4Swizzles,
};

use core::{
    iter::{Product, Sum},
    ops::*,
};

use gnum::{
    num::{Float, Real, ScalarReal},
    simd::{MaskLike, Select},
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// Creates a 3x3 matrix from three column vectors.
#[inline(always)]
#[must_use]
pub const fn gmat3<T: Real>(x_axis: GVec3<T>, y_axis: GVec3<T>, z_axis: GVec3<T>) -> GMat3<T> {
    GMat3::from_cols(x_axis, y_axis, z_axis)
}

/// A 3x3 column major matrix.
///
/// This 3x3 matrix type features convenience methods for creating and using linear and
/// affine transformations. If you are primarily dealing with 2D affine transformations the
/// [`Affine2`](crate::Affine2) type is much faster and more space efficient than
/// using a 3x3 matrix.
///
/// Linear transformations including 3D rotation and scale can be created using methods
/// such as [`Self::from_diagonal()`], [`Self::from_quat()`], [`Self::from_axis_angle()`],
/// [`Self::from_rotation_x()`], [`Self::from_rotation_y()`], or
/// [`Self::from_rotation_z()`].
///
/// The resulting matrices can be use to transform 3D vectors using regular vector
/// multiplication.
///
/// Affine transformations including 2D translation, rotation and scale can be created
/// using methods such as [`Self::from_translation()`], [`Self::from_angle()`],
/// [`Self::from_scale()`] and [`Self::from_scale_angle_translation()`].
///
/// The [`Self::transform_point2()`] and [`Self::transform_vector2()`] convenience methods
/// are provided for performing affine transforms on 2D vectors and points. These multiply
/// 2D inputs as 3D vectors with an implicit `z` value of `1` for points and `0` for
/// vectors respectively. These methods assume that `Self` contains a valid affine
/// transform.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[cfg_attr(
    feature = "zerocopy",
    derive(FromBytes, Immutable, IntoBytes, KnownLayout)
)]
#[repr(C)]
pub struct GMat3<T: Real> {
    /// The first column of the matrix.
    pub x_axis: GVec3<T>,
    /// The second column of the matrix.
    pub y_axis: GVec3<T>,
    /// The third column of the matrix.
    pub z_axis: GVec3<T>,
}

/// # Constants
impl<T: Real> GMat3<T> {
    /// All zeros.
    pub const ZERO: Self = Self::from_cols(GVec3::ZERO, GVec3::ZERO, GVec3::ZERO);

    /// The 3x3 identity matrix, where all diagonal elements are `1.0` and all off-diagonal elements are `0.0`.
    pub const IDENTITY: Self = Self::from_cols(GVec3::X, GVec3::Y, GVec3::Z);
}

impl<T: Float> GMat3<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(GVec3::NAN, GVec3::NAN, GVec3::NAN);
}

/// # Constructors
impl<T: Real> GMat3<T> {
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    #[must_use]
    const fn new(m00: T, m01: T, m02: T, m10: T, m11: T, m12: T, m20: T, m21: T, m22: T) -> Self {
        Self {
            x_axis: GVec3::new(m00, m01, m02),
            y_axis: GVec3::new(m10, m11, m12),
            z_axis: GVec3::new(m20, m21, m22),
        }
    }

    /// Creates a 3x3 matrix from three column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: GVec3<T>, y_axis: GVec3<T>, z_axis: GVec3<T>) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    /// Creates a 3x3 matrix from a `[T; 9]` array stored in column major order.
    /// If your data is stored in row major you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array(m: &[T; 9]) -> Self {
        Self::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
    }

    /// Creates a `[T; 9]` array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array(&self) -> [T; 9] {
        [
            self.x_axis.x,
            self.x_axis.y,
            self.x_axis.z,
            self.y_axis.x,
            self.y_axis.y,
            self.y_axis.z,
            self.z_axis.x,
            self.z_axis.y,
            self.z_axis.z,
        ]
    }

    /// Creates a 3x3 matrix from a `[[T; 3]; 3]` 2D array stored in column major order.
    /// If your data is in row major order you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[T; 3]; 3]) -> Self {
        Self::from_cols(
            GVec3::from_array(m[0]),
            GVec3::from_array(m[1]),
            GVec3::from_array(m[2]),
        )
    }

    /// Creates a `[[T; 3]; 3]` 2D array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[T; 3]; 3] {
        [
            self.x_axis.to_array(),
            self.y_axis.to_array(),
            self.z_axis.to_array(),
        ]
    }

    /// Creates a 3x3 matrix with its diagonal set to `diagonal` and all other entries set to 0.
    #[doc(alias = "scale")]
    #[inline]
    #[must_use]
    pub const fn from_diagonal(diagonal: GVec3<T>) -> Self {
        Self::new(
            diagonal.x,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            diagonal.y,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            diagonal.z,
        )
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
            x_axis: GVec3::select(boolean, if_true.x_axis, if_false.x_axis),
            y_axis: GVec3::select(boolean, if_true.y_axis, if_false.y_axis),
            z_axis: GVec3::select(boolean, if_true.z_axis, if_false.z_axis),
        }
    }

    /// Creates a 3x3 matrix from a 4x4 matrix, discarding the 4th row and column.
    #[inline]
    #[must_use]
    pub fn from_mat4(m: GMat4<T>) -> Self {
        Self::from_cols(m.x_axis.xyz(), m.y_axis.xyz(), m.z_axis.xyz())
    }

    /// Creates a 3x3 matrix from the minor of the given 4x4 matrix, discarding the `i`th column
    /// and `j`th row.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is greater than 3.
    #[inline]
    #[must_use]
    pub fn from_mat4_minor(m: GMat4<T>, i: usize, j: usize) -> Self {
        match (i, j) {
            (0, 0) => Self::from_cols(m.y_axis.yzw(), m.z_axis.yzw(), m.w_axis.yzw()),
            (0, 1) => Self::from_cols(m.y_axis.xzw(), m.z_axis.xzw(), m.w_axis.xzw()),
            (0, 2) => Self::from_cols(m.y_axis.xyw(), m.z_axis.xyw(), m.w_axis.xyw()),
            (0, 3) => Self::from_cols(m.y_axis.xyz(), m.z_axis.xyz(), m.w_axis.xyz()),
            (1, 0) => Self::from_cols(m.x_axis.yzw(), m.z_axis.yzw(), m.w_axis.yzw()),
            (1, 1) => Self::from_cols(m.x_axis.xzw(), m.z_axis.xzw(), m.w_axis.xzw()),
            (1, 2) => Self::from_cols(m.x_axis.xyw(), m.z_axis.xyw(), m.w_axis.xyw()),
            (1, 3) => Self::from_cols(m.x_axis.xyz(), m.z_axis.xyz(), m.w_axis.xyz()),
            (2, 0) => Self::from_cols(m.x_axis.yzw(), m.y_axis.yzw(), m.w_axis.yzw()),
            (2, 1) => Self::from_cols(m.x_axis.xzw(), m.y_axis.xzw(), m.w_axis.xzw()),
            (2, 2) => Self::from_cols(m.x_axis.xyw(), m.y_axis.xyw(), m.w_axis.xyw()),
            (2, 3) => Self::from_cols(m.x_axis.xyz(), m.y_axis.xyz(), m.w_axis.xyz()),
            (3, 0) => Self::from_cols(m.x_axis.yzw(), m.y_axis.yzw(), m.z_axis.yzw()),
            (3, 1) => Self::from_cols(m.x_axis.xzw(), m.y_axis.xzw(), m.z_axis.xzw()),
            (3, 2) => Self::from_cols(m.x_axis.xyw(), m.y_axis.xyw(), m.z_axis.xyw()),
            (3, 3) => Self::from_cols(m.x_axis.xyz(), m.y_axis.xyz(), m.z_axis.xyz()),
            _ => panic!("index out of bounds"),
        }
    }

    /// Creates a 3D rotation matrix from the given quaternion.
    #[inline]
    #[must_use]
    pub fn from_quat(rotation: GQuat<T>) -> Self {
        let x2 = rotation.x + rotation.x;
        let y2 = rotation.y + rotation.y;
        let z2 = rotation.z + rotation.z;
        let xx = rotation.x * x2;
        let xy = rotation.x * y2;
        let xz = rotation.x * z2;
        let yy = rotation.y * y2;
        let yz = rotation.y * z2;
        let zz = rotation.z * z2;
        let wx = rotation.w * x2;
        let wy = rotation.w * y2;
        let wz = rotation.w * z2;

        Self::from_cols(
            GVec3::new(T::ONE - (yy + zz), xy + wz, xz - wy),
            GVec3::new(xy - wz, T::ONE - (xx + zz), yz + wx),
            GVec3::new(xz + wy, yz - wx, T::ONE - (xx + yy)),
        )
    }

    /// Creates a 3D rotation matrix from a normalized rotation `axis` and `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: GVec3<T>, angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        let (xsin, ysin, zsin) = axis.mul(sin).into();
        let (x, y, z) = axis.into();
        let (x2, y2, z2) = axis.mul(axis).into();
        let omc = T::ONE - cos;
        let xyomc = x * y * omc;
        let xzomc = x * z * omc;
        let yzomc = y * z * omc;
        Self::from_cols(
            GVec3::new(x2 * omc + cos, xyomc + zsin, xzomc - ysin),
            GVec3::new(xyomc - zsin, y2 * omc + cos, yzomc + xsin),
            GVec3::new(xzomc + ysin, yzomc - xsin, z2 * omc + cos),
        )
    }
}

impl<T: ScalarReal> GMat3<T> {
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

impl<T: Real> GMat3<T> {
    /// Creates a 3D rotation matrix from `angle` (in radians) around the x axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: T) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::from_cols(
            GVec3::X,
            GVec3::new(T::ZERO, cosa, sina),
            GVec3::new(T::ZERO, -sina, cosa),
        )
    }

    /// Creates a 3D rotation matrix from `angle` (in radians) around the y axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: T) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::from_cols(
            GVec3::new(cosa, T::ZERO, -sina),
            GVec3::Y,
            GVec3::new(sina, T::ZERO, cosa),
        )
    }

    /// Creates a 3D rotation matrix from `angle` (in radians) around the z axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: T) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::from_cols(
            GVec3::new(cosa, sina, T::ZERO),
            GVec3::new(-sina, cosa, T::ZERO),
            GVec3::Z,
        )
    }

    /// Creates an affine transformation matrix from the given 2D `translation`.
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    #[must_use]
    pub fn from_translation(translation: GVec2<T>) -> Self {
        Self::from_cols(
            GVec3::X,
            GVec3::Y,
            GVec3::new(translation.x, translation.y, T::ONE),
        )
    }

    /// Creates an affine transformation matrix from the given 2D rotation `angle` (in
    /// radians).
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    #[must_use]
    pub fn from_angle(angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::from_cols(
            GVec3::new(cos, sin, T::ZERO),
            GVec3::new(-sin, cos, T::ZERO),
            GVec3::Z,
        )
    }

    /// Creates an affine transformation matrix from the given 2D `scale`, rotation `angle` (in
    /// radians) and `translation`.
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    #[must_use]
    pub fn from_scale_angle_translation(scale: GVec2<T>, angle: T, translation: GVec2<T>) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::from_cols(
            GVec3::new(cos * scale.x, sin * scale.x, T::ZERO),
            GVec3::new(-sin * scale.y, cos * scale.y, T::ZERO),
            GVec3::new(translation.x, translation.y, T::ONE),
        )
    }

    /// Creates an affine transformation matrix from the given non-uniform 2D `scale`.
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    #[must_use]
    pub fn from_scale(scale: GVec2<T>) -> Self {
        Self::from_cols(
            GVec3::new(scale.x, T::ZERO, T::ZERO),
            GVec3::new(T::ZERO, scale.y, T::ZERO),
            GVec3::Z,
        )
    }

    /// Creates an affine transformation matrix from the given 2x2 matrix.
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    pub fn from_mat2(m: GMat2<T>) -> Self {
        Self::from_cols(
            (m.x_axis, T::ZERO).into(),
            (m.y_axis, T::ZERO).into(),
            GVec3::Z,
        )
    }

    /// Creates a 3x3 matrix from the first 9 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 9 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice(slice: &[T]) -> Self {
        Self::new(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            slice[8],
        )
    }

    /// Writes the columns of `self` to the first 9 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 9 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [T]) {
        slice[0] = self.x_axis.x;
        slice[1] = self.x_axis.y;
        slice[2] = self.x_axis.z;
        slice[3] = self.y_axis.x;
        slice[4] = self.y_axis.y;
        slice[5] = self.y_axis.z;
        slice[6] = self.z_axis.x;
        slice[7] = self.z_axis.y;
        slice[8] = self.z_axis.z;
    }
}

/// # Operations
impl<T: Real> GMat3<T> {
    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 2.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> GVec3<T> {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns a mutable reference to the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 2.
    #[inline]
    pub fn col_mut(&mut self, index: usize) -> &mut GVec3<T> {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            2 => &mut self.z_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the matrix row for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 2.
    #[inline]
    #[must_use]
    pub fn row(&self, index: usize) -> GVec3<T> {
        match index {
            0 => GVec3::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            1 => GVec3::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            2 => GVec3::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            x_axis: GVec3::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            y_axis: GVec3::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            z_axis: GVec3::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
        }
    }

    /// Returns the diagonal of `self`.
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> GVec3<T> {
        GVec3::new(self.x_axis.x, self.y_axis.y, self.z_axis.z)
    }

    /// Returns the determinant of `self`.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> T {
        self.z_axis.dot(self.x_axis.cross(self.y_axis))
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
        let tmp0 = self.y_axis.cross(self.z_axis);
        let tmp1 = self.z_axis.cross(self.x_axis);
        let tmp2 = self.x_axis.cross(self.y_axis);
        let det = self.z_axis.dot(tmp2);
        let inv_det = GVec3::splat(det.recip());
        let inverted =
            Self::from_cols(tmp0.mul(inv_det), tmp1.mul(inv_det), tmp2.mul(inv_det)).transpose();
        if CHECKED {
            let invertible = det.num_ne(T::ZERO);
            (Self::select(invertible, inverted, Self::ZERO), invertible)
        } else {
            (inverted, T::Bool::TRUE)
        }
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

    /// Transforms the given 2D vector as a point.
    ///
    /// This is the equivalent of multiplying `rhs` as a 3D vector where `z` is `1`.
    ///
    /// This method assumes that `self` contains a valid affine transform.
    #[inline]
    #[must_use]
    pub fn transform_point2(&self, rhs: GVec2<T>) -> GVec2<T> {
        GMat2::from_cols(self.x_axis.xy(), self.y_axis.xy()) * rhs + self.z_axis.xy()
    }

    /// Rotates the given 2D vector.
    ///
    /// This is the equivalent of multiplying `rhs` as a 3D vector where `z` is `0`.
    ///
    /// This method assumes that `self` contains a valid affine transform.
    #[inline]
    #[must_use]
    pub fn transform_vector2(&self, rhs: GVec2<T>) -> GVec2<T> {
        GMat2::from_cols(self.x_axis.xy(), self.y_axis.xy()) * rhs
    }

    /// Creates a left-handed view matrix using a facing direction and an up direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(dir: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_rh(-dir, up)
    }

    /// Creates a right-handed view matrix using a facing direction and an up direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(dir: GVec3<T>, up: GVec3<T>) -> Self {
        let f = dir;
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self::from_cols(
            GVec3::new(s.x, u.x, -f.x),
            GVec3::new(s.y, u.y, -f.y),
            GVec3::new(s.z, u.z, -f.z),
        )
    }

    /// Creates a left-handed view matrix using a camera position, a focal point and an up
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: GVec3<T>, center: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_lh(center.sub(eye).normalize(), up)
    }

    /// Creates a right-handed view matrix using a camera position, a focal point and an up
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    pub fn look_at_rh(eye: GVec3<T>, center: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_rh(center.sub(eye).normalize(), up)
    }

    /// Transforms a 3D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec3(&self, rhs: GVec3<T>) -> GVec3<T> {
        let mut res = self.x_axis.mul(rhs.x);
        res = res.add(self.y_axis.mul(rhs.y));
        res = res.add(self.z_axis.mul(rhs.z));
        res
    }

    /// Transforms a 3D vector by the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn mul_transpose_vec3(&self, rhs: GVec3<T>) -> GVec3<T> {
        GVec3::new(
            self.x_axis.dot(rhs),
            self.y_axis.dot(rhs),
            self.z_axis.dot(rhs),
        )
    }

    /// Multiplies two 3x3 matrices.
    #[inline]
    #[must_use]
    pub fn mul_mat3(&self, rhs: &Self) -> Self {
        self.mul(rhs)
    }

    /// Adds two 3x3 matrices.
    #[inline]
    #[must_use]
    pub fn add_mat3(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }

    /// Subtracts two 3x3 matrices.
    #[inline]
    #[must_use]
    pub fn sub_mat3(&self, rhs: &Self) -> Self {
        self.sub(rhs)
    }

    /// Multiplies a 3x3 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn mul_scalar(&self, rhs: T) -> Self {
        Self::from_cols(
            self.x_axis.mul(rhs),
            self.y_axis.mul(rhs),
            self.z_axis.mul(rhs),
        )
    }

    /// Multiply `self` by a scaling vector `scale`.
    /// This is faster than creating a whole diagonal scaling matrix and then multiplying that.
    /// This operation is commutative.
    #[inline]
    #[must_use]
    pub fn mul_diagonal_scale(&self, scale: GVec3<T>) -> Self {
        Self::from_cols(
            self.x_axis * scale.x,
            self.y_axis * scale.y,
            self.z_axis * scale.z,
        )
    }

    /// Divides a 3x3 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn div_scalar(&self, rhs: T) -> Self {
        let rhs = GVec3::splat(rhs);
        Self::from_cols(
            self.x_axis.div(rhs),
            self.y_axis.div(rhs),
            self.z_axis.div(rhs),
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
    }

    /// Takes the absolute value of each element in `self`.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self::from_cols(self.x_axis.abs(), self.y_axis.abs(), self.z_axis.abs())
    }
}

impl<T: Float> GMat3<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> T::Bool {
        self.x_axis.is_finite() & self.y_axis.is_finite() & self.z_axis.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> T::Bool {
        self.x_axis.is_nan() | self.y_axis.is_nan() | self.z_axis.is_nan()
    }
}

impl<T: Real> Default for GMat3<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real + Add<Output = T>> Add for GMat3<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: GMat3<T>) -> Self {
        GMat3::from_cols(
            self.x_axis.add(rhs.x_axis),
            self.y_axis.add(rhs.y_axis),
            self.z_axis.add(rhs.z_axis),
        )
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat3<T>> for GMat3<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &GMat3<T>) -> Self {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<GMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: GMat3<T>) -> GMat3<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: &GMat3<T>) -> GMat3<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> AddAssign for GMat3<T> {
    #[inline]
    fn add_assign(&mut self, rhs: GMat3<T>) {
        *self = self.add(rhs);
    }
}

impl<T: Real + AddAssign> AddAssign<&GMat3<T>> for GMat3<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &GMat3<T>) {
        self.add_assign(*rhs);
    }
}

impl<T: Real + Sub<Output = T>> Sub for GMat3<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: GMat3<T>) -> Self {
        GMat3::from_cols(
            self.x_axis.sub(rhs.x_axis),
            self.y_axis.sub(rhs.y_axis),
            self.z_axis.sub(rhs.z_axis),
        )
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat3<T>> for GMat3<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &GMat3<T>) -> Self {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<GMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: GMat3<T>) -> GMat3<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: &GMat3<T>) -> GMat3<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> SubAssign for GMat3<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: GMat3<T>) {
        *self = self.sub(rhs);
    }
}

impl<T: Real + SubAssign> SubAssign<&GMat3<T>> for GMat3<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &GMat3<T>) {
        self.sub_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul for GMat3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: GMat3<T>) -> Self {
        GMat3::from_cols(
            self.x_axis.mul(rhs.x_axis),
            self.y_axis.mul(rhs.y_axis),
            self.z_axis.mul(rhs.z_axis),
        )
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat3<T>> for GMat3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &GMat3<T>) -> Self {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: GMat3<T>) -> GMat3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &GMat3<T>) -> GMat3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign for GMat3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GMat3<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + MulAssign> MulAssign<&GMat3<T>> for GMat3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GMat3<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec3<T>> for GMat3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: GVec3<T>) -> GVec3<T> {
        self.mul_vec3(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec3<T>> for GMat3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: &GVec3<T>) -> GVec3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec3<T>> for &GMat3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: GVec3<T>) -> GVec3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec3<T>> for &GMat3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: &GVec3<T>) -> GVec3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: T) -> GMat3<T> {
        self.mul_scalar(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GMat3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: T) -> GMat3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GMat3<T> {
        (*self).mul(*rhs)
    }
}

// We cannot implement scalar * matrix generically because of Rust's orphan rules.
macro_rules! impl_scalar_left_mul {
    ($($t:ty),*) => {
        $(
            impl Mul<GMat3<$t>> for $t {
                type Output = GMat3<$t>;
                #[inline]
                fn mul(self, rhs: GMat3<$t>) -> GMat3<$t> {
                    rhs.mul_scalar(self)
                }
            }

            impl Mul<&GMat3<$t>> for $t {
                type Output = GMat3<$t>;
                #[inline]
                fn mul(self, rhs: &GMat3<$t>) -> GMat3<$t> {
                    self.mul(*rhs)
                }
            }

            impl Mul<GMat3<$t>> for &$t {
                type Output = GMat3<$t>;
                #[inline]
                fn mul(self, rhs: GMat3<$t>) -> GMat3<$t> {
                    (*self).mul(rhs)
                }
            }

            impl Mul<&GMat3<$t>> for &$t {
                type Output = GMat3<$t>;
                #[inline]
                fn mul(self, rhs: &GMat3<$t>) -> GMat3<$t> {
                    (*self).mul(*rhs)
                }
            }
        )*
    };
}

// TODO: Implement for SIMD types
impl_scalar_left_mul!(f32, f64);

impl<T: Real + Mul<Output = T>> MulAssign<T> for GMat3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&T> for GMat3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Div<Output = T>> Div<T> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn div(self, rhs: T) -> GMat3<T> {
        self.div_scalar(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn div(self, rhs: &T) -> GMat3<T> {
        self.div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn div(self, rhs: T) -> GMat3<T> {
        (*self).div(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn div(self, rhs: &T) -> GMat3<T> {
        (*self).div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> DivAssign<T> for GMat3<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = self.div(rhs);
    }
}

impl<T: Real + Div<Output = T>> DivAssign<&T> for GMat3<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.div_assign(*rhs);
    }
}

impl<T: Real + Neg<Output = T>> Neg for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn neg(self) -> GMat3<T> {
        GMat3::from_cols(self.x_axis.neg(), self.y_axis.neg(), self.z_axis.neg())
    }
}

impl<T: Real + Neg<Output = T>> Neg for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn neg(self) -> GMat3<T> {
        (*self).neg()
    }
}

impl<T: Real> Sum<GMat3<T>> for GMat3<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, T: Real> Sum<&'a GMat3<T>> for GMat3<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl<T: Real> Product<GMat3<T>> for GMat3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl<'a, T: Real> Product<&'a GMat3<T>> for GMat3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| a * b)
    }
}

impl<T: Real> AsRef<[T; 9]> for GMat3<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 9] {
        unsafe { &*(self as *const GMat3<T> as *const [T; 9]) }
    }
}

impl<T: Real> AsMut<[T; 9]> for GMat3<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 9] {
        unsafe { &mut *(self as *mut GMat3<T> as *mut [T; 9]) }
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for GMat3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(p) = f.precision() {
            write!(
                f,
                "[{:.*}, {:.*}, {:.*}]",
                p, self.x_axis, p, self.y_axis, p, self.z_axis
            )
        } else {
            write!(f, "[{}, {}, {}]", self.x_axis, self.y_axis, self.z_axis)
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for GMat3<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(GMat3))
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis)
            .finish()
    }
}
