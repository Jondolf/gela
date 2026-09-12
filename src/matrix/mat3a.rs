use crate::matrix::{GMat2, GMat3, GMat4};
use crate::rotation::{EulerRot, GRot3};
use crate::vector::{GVec2, GVec3, Vec3A};

use core::{
    fmt,
    iter::{Product, Sum},
    ops::*,
};

use gnum::{
    cmp::NumEq,
    num::{NumCast, Real},
};

/// Creates a 3x3 column-major matrix with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Mat3`](crate::matrix::Mat3),
/// at the cost of a larger size.
#[inline(always)]
#[must_use]
pub const fn mat3a(x_axis: Vec3A, y_axis: Vec3A, z_axis: Vec3A) -> Mat3A {
    Mat3A::from_cols(x_axis, y_axis, z_axis)
}

/// A 3x3 column-major matrix with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Mat3`](crate::matrix::Mat3),
/// at the cost of a larger size.
///
/// This 3x3 matrix type features convenience methods for creating and using linear and
/// affine transformations. If you are primarily dealing with 2D affine transformations the
/// [`Affine2`](crate::affine::Affine2) type is much faster and more space efficient than
/// using a 3x3 matrix.
///
/// Linear transformations including 3D rotation and scale can be created using methods
/// such as [`Self::from_diagonal()`], [`Self::from_rotation()`], [`Self::from_axis_angle()`],
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
    derive(
        zerocopy_derive::FromBytes,
        zerocopy_derive::Immutable,
        zerocopy_derive::IntoBytes,
        zerocopy_derive::KnownLayout
    )
)]
#[repr(C)]
pub struct Mat3A {
    /// The first column of the matrix.
    pub x_axis: Vec3A,
    /// The second column of the matrix.
    pub y_axis: Vec3A,
    /// The third column of the matrix.
    pub z_axis: Vec3A,
}

/// # Constants
impl Mat3A {
    /// All zeros.
    pub const ZERO: Self = Self::from_cols(Vec3A::ZERO, Vec3A::ZERO, Vec3A::ZERO);

    /// The 3x3 identity matrix, where all diagonal elements are `1.0` and all off-diagonal elements are `0.0`.
    pub const IDENTITY: Self = Self::from_cols(Vec3A::X, Vec3A::Y, Vec3A::Z);

    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(Vec3A::NAN, Vec3A::NAN, Vec3A::NAN);
}

/// # Construction
impl Mat3A {
    /// Creates a 3x3 matrix from three column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: Vec3A, y_axis: Vec3A, z_axis: Vec3A) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    /// Creates a 3x3 matrix from a `[f32; 9]` array stored in column major order.
    /// If your data is stored in row major you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array(m: &[f32; 9]) -> Self {
        Self::from_cols(
            Vec3A::new(m[0], m[1], m[2]),
            Vec3A::new(m[3], m[4], m[5]),
            Vec3A::new(m[6], m[7], m[8]),
        )
    }

    /// Creates a `[f32; 9]` array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array(&self) -> [f32; 9] {
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

    /// Creates a 3x3 matrix from a `[[f32; 3]; 3]` 2D array stored in column major order.
    /// If your data is in row major order you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[f32; 3]; 3]) -> Self {
        Self::from_cols(
            Vec3A::from_array(m[0]),
            Vec3A::from_array(m[1]),
            Vec3A::from_array(m[2]),
        )
    }

    /// Creates a `[[f32; 3]; 3]` 2D array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array_2d(&self) -> [[f32; 3]; 3] {
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
    pub const fn from_diagonal(diagonal: GVec3<f32>) -> Self {
        Self::from_cols(
            Vec3A::new(diagonal.x, 0.0, 0.0),
            Vec3A::new(0.0, diagonal.y, 0.0),
            Vec3A::new(0.0, 0.0, diagonal.z),
        )
    }

    /// Creates a matrix from the elements in `if_true` and `if_false`, selecting which to use
    /// based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select(boolean: bool, if_true: Self, if_false: Self) -> Self {
        if boolean { if_true } else { if_false }
    }

    /// Creates a 3x3 matrix from a 4x4 matrix, discarding the 4th row and column.
    #[inline]
    #[must_use]
    pub fn from_mat4(m: GMat4<f32>) -> Self {
        GMat3::from_mat4(m).into()
    }

    /// Creates a 3x3 matrix from the minor of the given 4x4 matrix, discarding the `i`th column
    /// and `j`th row.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is greater than 3.
    #[inline]
    #[must_use]
    pub fn from_mat4_minor(m: GMat4<f32>, i: usize, j: usize) -> Self {
        GMat3::from_mat4_minor(m, i, j).into()
    }

    /// Creates a 3D rotation matrix from the given quaternion.
    #[inline]
    #[must_use]
    pub fn from_rotation(rotation: GRot3<f32>) -> Self {
        GMat3::from_rotation(rotation).into()
    }

    /// Creates a 3D rotation matrix from `angle` (in radians) around the x axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        GMat3::from_rotation_x(angle).into()
    }

    /// Creates a 3D rotation matrix from `angle` (in radians) around the y axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        GMat3::from_rotation_y(angle).into()
    }

    /// Creates a 3D rotation matrix from `angle` (in radians) around the z axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        GMat3::from_rotation_z(angle).into()
    }

    /// Creates a 3D rotation matrix from a normalized rotation `axis` and `angle` (in radians).
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: GVec3<f32>, angle: f32) -> Self {
        GMat3::from_axis_angle(axis, angle).into()
    }

    /// Creates a 3D rotation matrix from the given euler rotation sequence and the angles (in radians).
    #[inline]
    #[must_use]
    pub fn from_euler(order: EulerRot, a: f32, b: f32, c: f32) -> Self {
        GMat3::from_euler(order, a, b, c).into()
    }

    /// Extract Euler angles with the given Euler rotation order.
    ///
    /// Note if the input matrix contains scales, shears, or other non-rotation transformations then
    /// the resulting Euler angles will be ill-defined.
    #[inline]
    #[must_use]
    pub fn to_euler(&self, order: EulerRot) -> (f32, f32, f32) {
        self.to_mat3().to_euler(order)
    }

    /// Creates an affine transformation matrix from the given 2D `translation`.
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    #[must_use]
    pub fn from_translation(translation: GVec2<f32>) -> Self {
        GMat3::from_translation(translation).into()
    }

    /// Creates an affine transformation matrix from the given 2D rotation `angle` (in
    /// radians).
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    #[must_use]
    pub fn from_angle(angle: f32) -> Self {
        GMat3::from_angle(angle).into()
    }

    /// Creates an affine transformation matrix from the given 2D `scale`, rotation `angle` (in
    /// radians) and `translation`.
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    #[must_use]
    pub fn from_scale_angle_translation(
        scale: GVec2<f32>,
        angle: f32,
        translation: GVec2<f32>,
    ) -> Self {
        GMat3::from_scale_angle_translation(scale, angle, translation).into()
    }

    /// Creates an affine transformation matrix from the given non-uniform 2D `scale`.
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    #[must_use]
    pub fn from_scale(scale: GVec2<f32>) -> Self {
        GMat3::from_scale(scale).into()
    }

    /// Creates an affine transformation matrix from the given 2x2 matrix.
    ///
    /// The resulting matrix can be used to transform 2D points and vectors. See
    /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
    #[inline]
    pub fn from_mat2(m: GMat2<f32>) -> Self {
        GMat3::from_mat2(m).into()
    }

    /// Creates a 3x3 matrix from the first 9 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 9 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice(slice: &[f32]) -> Self {
        Self::from_cols(
            Vec3A::new(slice[0], slice[1], slice[2]),
            Vec3A::new(slice[3], slice[4], slice[5]),
            Vec3A::new(slice[6], slice[7], slice[8]),
        )
    }

    /// Writes the columns of `self` to the first 9 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 9 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [f32]) {
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
impl Mat3A {
    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 2.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> Vec3A {
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
    pub fn col_mut(&mut self, index: usize) -> &mut Vec3A {
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
    pub fn row(&self, index: usize) -> Vec3A {
        match index {
            0 => Vec3A::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            1 => Vec3A::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            2 => Vec3A::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            x_axis: Vec3A::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            y_axis: Vec3A::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            z_axis: Vec3A::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
        }
    }

    /// Returns the diagonal of `self`.
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> GVec3<f32> {
        GVec3::new(self.x_axis.x, self.y_axis.y, self.z_axis.z)
    }

    /// Returns the determinant of `self`.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> f32 {
        self.z_axis.dot(self.x_axis.cross(self.y_axis))
    }

    /// Returns the inverse of `self`.
    ///
    /// If the matrix is not invertible the returned matrix will be invalid.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let tmp0 = self.y_axis.cross(self.z_axis);
        let tmp1 = self.z_axis.cross(self.x_axis);
        let tmp2 = self.x_axis.cross(self.y_axis);
        let inv_det = Vec3A::splat(Real::recip(self.z_axis.dot(tmp2)));
        Self::from_cols(tmp0 * inv_det, tmp1 * inv_det, tmp2 * inv_det).transpose()
    }

    /// Returns the inverse of `self` or `Mat3A::ZERO` if the matrix is not invertible.
    #[inline]
    #[must_use]
    pub fn inverse_or_zero(&self) -> Self {
        let tmp0 = self.y_axis.cross(self.z_axis);
        let tmp1 = self.z_axis.cross(self.x_axis);
        let tmp2 = self.x_axis.cross(self.y_axis);
        let det = self.z_axis.dot(tmp2);
        if det.num_ne(0.0) {
            let inv_det = Vec3A::splat(Real::recip(det));
            Self::from_cols(tmp0 * inv_det, tmp1 * inv_det, tmp2 * inv_det).transpose()
        } else {
            Self::ZERO
        }
    }

    /// Transforms the given 2D vector as a point.
    ///
    /// This is the equivalent of multiplying `rhs` as a 3D vector where `z` is `1`.
    ///
    /// This method assumes that `self` contains a valid affine transform.
    #[inline]
    #[must_use]
    pub fn transform_point2(&self, rhs: GVec2<f32>) -> GVec2<f32> {
        self.to_mat3().transform_point2(rhs)
    }

    /// Rotates the given 2D vector.
    ///
    /// This is the equivalent of multiplying `rhs` as a 3D vector where `z` is `0`.
    ///
    /// This method assumes that `self` contains a valid affine transform.
    #[inline]
    #[must_use]
    pub fn transform_vector2(&self, rhs: GVec2<f32>) -> GVec2<f32> {
        self.to_mat3().transform_vector2(rhs)
    }

    /// Creates a left-handed view matrix using a facing direction and an up direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(dir: GVec3<f32>, up: GVec3<f32>) -> Self {
        GMat3::look_to_lh(dir, up).into()
    }

    /// Creates a right-handed view matrix using a facing direction and an up direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(dir: GVec3<f32>, up: GVec3<f32>) -> Self {
        GMat3::look_to_rh(dir, up).into()
    }

    /// Creates a left-handed view matrix using a camera position, a focal point and an up
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: GVec3<f32>, center: GVec3<f32>, up: GVec3<f32>) -> Self {
        GMat3::look_at_lh(eye, center, up).into()
    }

    /// Creates a right-handed view matrix using a camera position, a focal point and an up
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    pub fn look_at_rh(eye: GVec3<f32>, center: GVec3<f32>, up: GVec3<f32>) -> Self {
        GMat3::look_at_rh(eye, center, up).into()
    }

    /// Transforms a 3D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec3(&self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.to_mat3().mul_vec3(rhs)
    }

    /// Transforms a SIMD-aligned 3D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec3a(&self, rhs: Vec3A) -> Vec3A {
        let mut res = self.x_axis * rhs.x;
        res += self.y_axis * rhs.y;
        res += self.z_axis * rhs.z;
        res
    }

    /// Transforms a 3D vector by the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn mul_transpose_vec3(&self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.to_mat3().mul_transpose_vec3(rhs)
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
    pub fn mul_scalar(&self, rhs: f32) -> Self {
        Self::from_cols(self.x_axis * rhs, self.y_axis * rhs, self.z_axis * rhs)
    }

    /// Multiply `self` by a scaling vector `scale`.
    /// This is faster than creating a whole diagonal scaling matrix and then multiplying that.
    /// This operation is commutative.
    #[inline]
    #[must_use]
    pub fn mul_diagonal_scale(&self, scale: GVec3<f32>) -> Self {
        Self::from_cols(
            self.x_axis * scale.x,
            self.y_axis * scale.y,
            self.z_axis * scale.z,
        )
    }

    /// Divides a 3x3 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn div_scalar(&self, rhs: f32) -> Self {
        Self::from_cols(self.x_axis / rhs, self.y_axis / rhs, self.z_axis / rhs)
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
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: f32) -> bool {
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

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite() & self.y_axis.is_finite() & self.z_axis.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() | self.y_axis.is_nan() | self.z_axis.is_nan()
    }
}

/// # Conversions
impl Mat3A {
    /// Creates a [`Mat3A`] from a [`GMat3<f32>`].
    #[inline(always)]
    #[must_use]
    pub const fn from_mat3(m: GMat3<f32>) -> Self {
        Self::from_cols(
            Vec3A::from_vec3(m.x_axis),
            Vec3A::from_vec3(m.y_axis),
            Vec3A::from_vec3(m.z_axis),
        )
    }

    /// Converts `self` to a [`GMat3<f32>`].
    #[inline(always)]
    #[must_use]
    pub fn to_mat3(self) -> GMat3<f32> {
        GMat3::from_cols(
            self.x_axis.to_vec3(),
            self.y_axis.to_vec3(),
            self.z_axis.to_vec3(),
        )
    }

    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GMat3<U>
    where
        f32: NumCast<U>,
    {
        self.to_mat3().cast()
    }
}

impl Default for Mat3A {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl fmt::Debug for Mat3A {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Mat3A")
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis)
            .finish()
    }
}

impl fmt::Display for Mat3A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl Add for Mat3A {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_cols(
            self.x_axis + rhs.x_axis,
            self.y_axis + rhs.y_axis,
            self.z_axis + rhs.z_axis,
        )
    }
}

impl Add<&Mat3A> for Mat3A {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Mat3A) -> Self {
        self.add(*rhs)
    }
}

impl AddAssign for Mat3A {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs);
    }
}

impl Sub for Mat3A {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_cols(
            self.x_axis - rhs.x_axis,
            self.y_axis - rhs.y_axis,
            self.z_axis - rhs.z_axis,
        )
    }
}

impl Sub<&Mat3A> for Mat3A {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Mat3A) -> Self {
        self.sub(*rhs)
    }
}

impl SubAssign for Mat3A {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(rhs);
    }
}

impl Mul for Mat3A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_cols(
            self.mul_vec3a(rhs.x_axis),
            self.mul_vec3a(rhs.y_axis),
            self.mul_vec3a(rhs.z_axis),
        )
    }
}

impl Mul<&Mat3A> for Mat3A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Mat3A) -> Self {
        self.mul(*rhs)
    }
}

impl MulAssign for Mat3A {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs);
    }
}

impl Mul<GVec3<f32>> for Mat3A {
    type Output = GVec3<f32>;
    #[inline]
    fn mul(self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.mul_vec3(rhs)
    }
}

impl Mul<Vec3A> for Mat3A {
    type Output = Vec3A;
    #[inline]
    fn mul(self, rhs: Vec3A) -> Vec3A {
        self.mul_vec3a(rhs)
    }
}

impl Mul<f32> for Mat3A {
    type Output = Mat3A;
    #[inline]
    fn mul(self, rhs: f32) -> Mat3A {
        self.mul_scalar(rhs)
    }
}

impl Mul<Mat3A> for f32 {
    type Output = Mat3A;
    #[inline]
    fn mul(self, rhs: Mat3A) -> Mat3A {
        rhs.mul_scalar(self)
    }
}

impl MulAssign<f32> for Mat3A {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = self.mul(rhs);
    }
}

impl Div<f32> for Mat3A {
    type Output = Mat3A;
    #[inline]
    fn div(self, rhs: f32) -> Mat3A {
        self.div_scalar(rhs)
    }
}

impl DivAssign<f32> for Mat3A {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self = self.div(rhs);
    }
}

impl Neg for Mat3A {
    type Output = Mat3A;
    #[inline]
    fn neg(self) -> Mat3A {
        Self::from_cols(-self.x_axis, -self.y_axis, -self.z_axis)
    }
}

impl Sum for Mat3A {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl Product for Mat3A {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl From<GMat3<f32>> for Mat3A {
    #[inline(always)]
    fn from(m: GMat3<f32>) -> Self {
        Self::from_mat3(m)
    }
}

impl From<Mat3A> for GMat3<f32> {
    #[inline(always)]
    fn from(m: Mat3A) -> Self {
        m.to_mat3()
    }
}
