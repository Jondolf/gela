use crate::matrix::GMat3;
use crate::matrix::GMat4;
use crate::rotation::{EulerRot, GRot3};
use crate::vector::{GVec3, GVec4, Vec4A};

use core::{
    fmt,
    iter::{Product, Sum},
    ops::*,
};

use gnum::num::{NumCast, Real};

/// Creates a 4x4 column-major matrix with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Mat4`](crate::matrix::Mat4).
#[inline(always)]
#[must_use]
pub const fn mat4a(x_axis: Vec4A, y_axis: Vec4A, z_axis: Vec4A, w_axis: Vec4A) -> Mat4A {
    Mat4A::from_cols(x_axis, y_axis, z_axis, w_axis)
}

/// A 4x4 column-major matrix with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Mat4`](crate::matrix::Mat4).
///
/// This 4x4 matrix type features convenience methods for creating and using affine transforms and
/// perspective projections. If you are primarily dealing with 3D affine transformations
/// considering using [`Affine3A`](crate::affine::Affine3A) which is faster than a 4x4 matrix
/// for some affine operations.
///
/// Affine transformations including 3D translation, rotation and scale can be created
/// using methods such as [`Self::from_translation()`], [`Self::from_rotation()`],
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
    derive(
        zerocopy_derive::FromBytes,
        zerocopy_derive::Immutable,
        zerocopy_derive::IntoBytes,
        zerocopy_derive::KnownLayout
    )
)]
#[repr(C)]
#[repr(align(16))]
pub struct Mat4A {
    /// The first column of the matrix.
    pub x_axis: Vec4A,
    /// The second column of the matrix.
    pub y_axis: Vec4A,
    /// The third column of the matrix.
    pub z_axis: Vec4A,
    /// The fourth column of the matrix.
    pub w_axis: Vec4A,
}

/// # Constants
impl Mat4A {
    /// All zeros.
    pub const ZERO: Self = Self::from_cols(Vec4A::ZERO, Vec4A::ZERO, Vec4A::ZERO, Vec4A::ZERO);

    /// The 4x4 identity matrix, where all diagonal elements are `1.0` and all off-diagonal elements are `0.0`.
    pub const IDENTITY: Self = Self::from_cols(Vec4A::X, Vec4A::Y, Vec4A::Z, Vec4A::W);

    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(Vec4A::NAN, Vec4A::NAN, Vec4A::NAN, Vec4A::NAN);
}

/// # Construction
impl Mat4A {
    /// Creates a 4x4 matrix from four column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: Vec4A, y_axis: Vec4A, z_axis: Vec4A, w_axis: Vec4A) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    /// Creates a 4x4 matrix from a `[f32; 16]` array stored in column major order.
    /// If your data is stored in row major you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array(m: &[f32; 16]) -> Self {
        Self::from_cols(
            Vec4A::new(m[0], m[1], m[2], m[3]),
            Vec4A::new(m[4], m[5], m[6], m[7]),
            Vec4A::new(m[8], m[9], m[10], m[11]),
            Vec4A::new(m[12], m[13], m[14], m[15]),
        )
    }

    /// Creates a `[f32; 16]` array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array(&self) -> [f32; 16] {
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

    /// Creates a 4x4 matrix from a `[[f32; 4]; 4]` 2D array stored in column major order.
    /// If your data is in row major order you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[f32; 4]; 4]) -> Self {
        Self::from_cols(
            Vec4A::from_array(m[0]),
            Vec4A::from_array(m[1]),
            Vec4A::from_array(m[2]),
            Vec4A::from_array(m[3]),
        )
    }

    /// Creates a `[[f32; 4]; 4]` 2D array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array_2d(&self) -> [[f32; 4]; 4] {
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
    pub const fn from_diagonal(diagonal: GVec4<f32>) -> Self {
        Self::from_cols(
            Vec4A::new(diagonal.x, 0.0, 0.0, 0.0),
            Vec4A::new(0.0, diagonal.y, 0.0, 0.0),
            Vec4A::new(0.0, 0.0, diagonal.z, 0.0),
            Vec4A::new(0.0, 0.0, 0.0, diagonal.w),
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

    /// Creates an affine transformation matrix from the given 3D `scale`, `rotation`, and `translation`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: GVec3<f32>,
        rotation: GRot3<f32>,
        translation: GVec3<f32>,
    ) -> Self {
        GMat4::from_scale_rotation_translation(scale, rotation, translation).into()
    }

    /// Creates an affine transformation matrix from the given 3D `translation`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_translation(rotation: GRot3<f32>, translation: GVec3<f32>) -> Self {
        GMat4::from_rotation_translation(rotation, translation).into()
    }

    /// Extracts `scale`, `rotation` and `translation` from `self`. The input matrix is
    /// expected to be a 3D affine transformation matrix otherwise the output will be invalid.
    #[inline]
    #[must_use]
    pub fn to_scale_rotation_translation(&self) -> (GVec3<f32>, GRot3<f32>, GVec3<f32>) {
        self.to_mat4().to_scale_rotation_translation()
    }

    /// Creates an affine transformation matrix from the given 3x3 linear transformation
    /// matrix.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_mat3(m: GMat3<f32>) -> Self {
        GMat4::from_mat3(m).into()
    }

    /// Creates an affine transformation matrics from a 3x3 matrix (expressing scale, shear and
    /// rotation) and a translation vector.
    ///
    /// Equivalent to `Mat4A::from_translation(translation) * Mat4A::from_mat3(mat3)`
    #[inline]
    #[must_use]
    pub fn from_mat3_translation(mat3: GMat3<f32>, translation: GVec3<f32>) -> Self {
        GMat4::from_mat3_translation(mat3, translation).into()
    }

    /// Creates an affine transformation matrix from the given 3D `translation`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_translation(translation: GVec3<f32>) -> Self {
        GMat4::from_translation(translation).into()
    }

    /// Creates an affine transformation matrix from the given `rotation` quaternion.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation(rotation: GRot3<f32>) -> Self {
        GMat4::from_rotation(rotation).into()
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the x axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        GMat4::from_rotation_x(angle).into()
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the y axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        GMat4::from_rotation_y(angle).into()
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the z axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        GMat4::from_rotation_z(angle).into()
    }

    /// Creates an affine transformation matrix containing a 3D rotation around a normalized
    /// rotation `axis` of `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: GVec3<f32>, angle: f32) -> Self {
        GMat4::from_axis_angle(axis, angle).into()
    }

    /// Creates a 3D rotation matrix from the given euler rotation sequence and the angles (in radians).
    #[inline]
    #[must_use]
    pub fn from_euler(order: EulerRot, a: f32, b: f32, c: f32) -> Self {
        GMat4::from_euler(order, a, b, c).into()
    }

    /// Extract Euler angles with the given Euler rotation order.
    ///
    /// Note that if the input matrix contains scales, shears, or other non-rotation
    /// transformations, the output of this function is ill-defined.
    #[inline]
    #[must_use]
    pub fn to_euler(&self, order: EulerRot) -> (f32, f32, f32) {
        self.to_mat4().to_euler(order)
    }

    /// Creates an affine transformation matrix containing the given 3D non-uniform `scale`.
    ///
    /// The resulting matrix can be used to transform 3D points and vectors.
    /// See [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn from_scale(scale: GVec3<f32>) -> Self {
        GMat4::from_scale(scale).into()
    }

    /// Creates a 4x4 matrix from the first 16 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 16 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice(slice: &[f32]) -> Self {
        Self::from_cols(
            Vec4A::new(slice[0], slice[1], slice[2], slice[3]),
            Vec4A::new(slice[4], slice[5], slice[6], slice[7]),
            Vec4A::new(slice[8], slice[9], slice[10], slice[11]),
            Vec4A::new(slice[12], slice[13], slice[14], slice[15]),
        )
    }

    /// Writes the columns of `self` to the first 16 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 16 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [f32]) {
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

/// # Projections
impl Mat4A {
    /// Creates a left-handed view matrix from a camera position, a facing direction, and an up direction.
    ///
    /// This is for a left-handed coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(eye: GVec3<f32>, dir: GVec3<f32>, up: GVec3<f32>) -> Self {
        GMat4::look_to_lh(eye, dir, up).into()
    }

    /// Creates a right-handed view matrix from a camera position, a facing direction, and an up direction.
    ///
    /// This is for a right-handed coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(eye: GVec3<f32>, dir: GVec3<f32>, up: GVec3<f32>) -> Self {
        GMat4::look_to_rh(eye, dir, up).into()
    }

    /// Creates a left-handed view matrix using a camera position, a focal point, and an up direction.
    ///
    /// This is for a left-handed coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: GVec3<f32>, center: GVec3<f32>, up: GVec3<f32>) -> Self {
        GMat4::look_at_lh(eye, center, up).into()
    }

    /// Creates a right-handed view matrix using a camera position, a focal point, and an up direction.
    ///
    /// This is for a right-handed coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    pub fn look_at_rh(eye: GVec3<f32>, center: GVec3<f32>, up: GVec3<f32>) -> Self {
        GMat4::look_at_rh(eye, center, up).into()
    }

    /// Creates a right-handed perspective projection matrix with [-1,1] depth range.
    ///
    /// This is the same as the OpenGL `glFrustum` function.
    ///
    /// See <https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml>
    #[inline]
    #[must_use]
    pub fn frustum_rh_gl(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        GMat4::frustum_rh_gl(left, right, bottom, top, z_near, z_far).into()
    }

    /// Creates a left-handed perspective projection matrix with `[0,1]` depth range.
    #[inline]
    #[must_use]
    pub fn frustum_lh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        GMat4::frustum_lh(left, right, bottom, top, z_near, z_far).into()
    }

    /// Creates a right-handed perspective projection matrix with `[0,1]` depth range.
    #[inline]
    #[must_use]
    pub fn frustum_rh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        GMat4::frustum_rh(left, right, bottom, top, z_near, z_far).into()
    }

    /// Creates a right-handed perspective projection matrix with `[-1,1]` depth range.
    ///
    /// Useful to map the standard right-handed coordinate system into what OpenGL expects.
    ///
    /// This is the same as the OpenGL `gluPerspective` function.
    /// See <https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml>
    #[inline]
    #[must_use]
    pub fn perspective_rh_gl(
        fov_y_radians: f32,
        aspect_ratio: f32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        GMat4::perspective_rh_gl(fov_y_radians, aspect_ratio, z_near, z_far).into()
    }

    /// Creates a left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Useful to map the standard left-handed coordinate system into what WebGPU/Metal/Direct3D expect.
    #[inline]
    #[must_use]
    pub fn perspective_lh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        GMat4::perspective_lh(fov_y_radians, aspect_ratio, z_near, z_far).into()
    }

    /// Creates a right-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Useful to map the standard right-handed coordinate system into what WebGPU/Metal/Direct3D expect.
    #[inline]
    #[must_use]
    pub fn perspective_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        GMat4::perspective_rh(fov_y_radians, aspect_ratio, z_near, z_far).into()
    }

    /// Creates an infinite left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Like `perspective_lh`, but with an infinite value for `z_far`.
    /// The result is that points near `z_near` are mapped to depth `0`, and as they move towards infinity the depth approaches `1`.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_lh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Self {
        GMat4::perspective_infinite_lh(fov_y_radians, aspect_ratio, z_near).into()
    }

    /// Creates an infinite reverse left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Similar to `perspective_infinite_lh`, but maps `Z = z_near` to a depth of `1` and `Z = infinity` to a depth of `0`.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_reverse_lh(
        fov_y_radians: f32,
        aspect_ratio: f32,
        z_near: f32,
    ) -> Self {
        GMat4::perspective_infinite_reverse_lh(fov_y_radians, aspect_ratio, z_near).into()
    }

    /// Creates an infinite right-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Like `perspective_rh`, but with an infinite value for `z_far`.
    /// The result is that points near `z_near` are mapped to depth `0`, and as they move towards infinity the depth approaches `1`.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Self {
        GMat4::perspective_infinite_rh(fov_y_radians, aspect_ratio, z_near).into()
    }

    /// Creates an infinite reverse right-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// Similar to `perspective_infinite_rh`, but maps `Z = z_near` to a depth of `1` and `Z = infinity` to a depth of `0`.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_reverse_rh(
        fov_y_radians: f32,
        aspect_ratio: f32,
        z_near: f32,
    ) -> Self {
        GMat4::perspective_infinite_reverse_rh(fov_y_radians, aspect_ratio, z_near).into()
    }

    /// Creates a right-handed orthographic projection matrix with `[-1,1]` depth
    /// range.  This is the same as the OpenGL `glOrtho` function in OpenGL.
    /// See
    /// <https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml>
    ///
    /// Useful to map a right-handed coordinate system to the normalized device coordinates that OpenGL expects.
    #[inline]
    #[must_use]
    pub fn orthographic_rh_gl(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        GMat4::orthographic_rh_gl(left, right, bottom, top, near, far).into()
    }

    /// Creates a left-handed orthographic projection matrix with `[0,1]` depth range.
    ///
    /// Useful to map a left-handed coordinate system to the normalized device coordinates that WebGPU/Direct3D/Metal expect.
    #[inline]
    #[must_use]
    pub fn orthographic_lh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        GMat4::orthographic_lh(left, right, bottom, top, near, far).into()
    }

    /// Creates a right-handed orthographic projection matrix with `[0,1]` depth range.
    ///
    /// Useful to map a right-handed coordinate system to the normalized device coordinates that WebGPU/Direct3D/Metal expect.
    #[inline]
    #[must_use]
    pub fn orthographic_rh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        GMat4::orthographic_rh(left, right, bottom, top, near, far).into()
    }
}

/// # Operations
/// Two-input lane shuffle, equivalent to x86's `_mm_shuffle_ps(a, b, ..)`: the result is
/// `[a[IA], a[IB], b[IC], b[ID]]`. With compile-time indices this lowers to a single `shufps`,
/// which is why the determinant and inverse routines below reach for it instead of building
/// vectors component by component.
#[inline(always)]
#[must_use]
fn shuffle<const IA: usize, const IB: usize, const IC: usize, const ID: usize>(
    a: Vec4A,
    b: Vec4A,
) -> Vec4A {
    Vec4A::new(a[IA], a[IB], b[IC], b[ID])
}

impl Mat4A {
    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> Vec4A {
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
    pub fn col_mut(&mut self, index: usize) -> &mut Vec4A {
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
    pub fn row(&self, index: usize) -> Vec4A {
        match index {
            0 => Vec4A::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
            1 => Vec4A::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
            2 => Vec4A::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
            3 => Vec4A::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            x_axis: Vec4A::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
            y_axis: Vec4A::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
            z_axis: Vec4A::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
            w_axis: Vec4A::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
        }
    }

    /// Returns the diagonal of `self`.
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> GVec4<f32> {
        GVec4::new(self.x_axis.x, self.y_axis.y, self.z_axis.z, self.w_axis.w)
    }

    /// Returns the determinant of `self`.
    #[must_use]
    pub fn determinant(&self) -> f32 {
        // The scalar [`Mat4`](crate::matrix::Mat4) cofactor expansion already vectorizes well and,
        // measured on this workload, beats a SIMD `_mm_shuffle_ps`-based determinant, so there is
        // nothing to gain by reimplementing it on the columns here. (The inverse below is a
        // different story — see [`Self::inverse`].)
        self.to_mat4().determinant()
    }

    /// Returns the inverse of `self`, or [`Mat4A::ZERO`] if the matrix is not invertible.
    ///
    /// This is bit-identical to [`Mat4::inverse`](crate::matrix::Mat4::inverse), but computed
    /// directly on the SIMD-aligned columns instead of round-tripping through the scalar
    /// [`Mat4`](crate::matrix::Mat4).
    #[inline(always)]
    #[must_use]
    fn inverse_impl(&self) -> Self {
        // Based on <https://github.com/g-truc/glm> `glm_mat4_inverse`. Each `shuffle` is a single
        // `shufps`, so the whole cofactor computation stays in vector registers.
        let (x, y, z, w) = (self.x_axis, self.y_axis, self.z_axis, self.w_axis);

        let fac0 = {
            let swp0a = shuffle::<3, 3, 3, 3>(w, z);
            let swp0b = shuffle::<2, 2, 2, 2>(w, z);
            let swp00 = shuffle::<2, 2, 2, 2>(z, y);
            let swp01 = shuffle::<0, 0, 0, 2>(swp0a, swp0a);
            let swp02 = shuffle::<0, 0, 0, 2>(swp0b, swp0b);
            let swp03 = shuffle::<3, 3, 3, 3>(z, y);
            swp00 * swp01 - swp02 * swp03
        };
        let fac1 = {
            let swp0a = shuffle::<3, 3, 3, 3>(w, z);
            let swp0b = shuffle::<1, 1, 1, 1>(w, z);
            let swp00 = shuffle::<1, 1, 1, 1>(z, y);
            let swp01 = shuffle::<0, 0, 0, 2>(swp0a, swp0a);
            let swp02 = shuffle::<0, 0, 0, 2>(swp0b, swp0b);
            let swp03 = shuffle::<3, 3, 3, 3>(z, y);
            swp00 * swp01 - swp02 * swp03
        };
        let fac2 = {
            let swp0a = shuffle::<2, 2, 2, 2>(w, z);
            let swp0b = shuffle::<1, 1, 1, 1>(w, z);
            let swp00 = shuffle::<1, 1, 1, 1>(z, y);
            let swp01 = shuffle::<0, 0, 0, 2>(swp0a, swp0a);
            let swp02 = shuffle::<0, 0, 0, 2>(swp0b, swp0b);
            let swp03 = shuffle::<2, 2, 2, 2>(z, y);
            swp00 * swp01 - swp02 * swp03
        };
        let fac3 = {
            let swp0a = shuffle::<3, 3, 3, 3>(w, z);
            let swp0b = shuffle::<0, 0, 0, 0>(w, z);
            let swp00 = shuffle::<0, 0, 0, 0>(z, y);
            let swp01 = shuffle::<0, 0, 0, 2>(swp0a, swp0a);
            let swp02 = shuffle::<0, 0, 0, 2>(swp0b, swp0b);
            let swp03 = shuffle::<3, 3, 3, 3>(z, y);
            swp00 * swp01 - swp02 * swp03
        };
        let fac4 = {
            let swp0a = shuffle::<2, 2, 2, 2>(w, z);
            let swp0b = shuffle::<0, 0, 0, 0>(w, z);
            let swp00 = shuffle::<0, 0, 0, 0>(z, y);
            let swp01 = shuffle::<0, 0, 0, 2>(swp0a, swp0a);
            let swp02 = shuffle::<0, 0, 0, 2>(swp0b, swp0b);
            let swp03 = shuffle::<2, 2, 2, 2>(z, y);
            swp00 * swp01 - swp02 * swp03
        };
        let fac5 = {
            let swp0a = shuffle::<1, 1, 1, 1>(w, z);
            let swp0b = shuffle::<0, 0, 0, 0>(w, z);
            let swp00 = shuffle::<0, 0, 0, 0>(z, y);
            let swp01 = shuffle::<0, 0, 0, 2>(swp0a, swp0a);
            let swp02 = shuffle::<0, 0, 0, 2>(swp0b, swp0b);
            let swp03 = shuffle::<1, 1, 1, 1>(z, y);
            swp00 * swp01 - swp02 * swp03
        };

        let sign_a = Vec4A::new(-1.0, 1.0, -1.0, 1.0);
        let sign_b = Vec4A::new(1.0, -1.0, 1.0, -1.0);

        let tmp0 = shuffle::<0, 0, 0, 0>(y, x);
        let vec0 = shuffle::<0, 2, 2, 2>(tmp0, tmp0);
        let tmp1 = shuffle::<1, 1, 1, 1>(y, x);
        let vec1 = shuffle::<0, 2, 2, 2>(tmp1, tmp1);
        let tmp2 = shuffle::<2, 2, 2, 2>(y, x);
        let vec2 = shuffle::<0, 2, 2, 2>(tmp2, tmp2);
        let tmp3 = shuffle::<3, 3, 3, 3>(y, x);
        let vec3 = shuffle::<0, 2, 2, 2>(tmp3, tmp3);

        let inv0 = sign_b * (vec1 * fac0 - vec2 * fac1 + vec3 * fac2);
        let inv1 = sign_a * (vec0 * fac0 - vec2 * fac3 + vec3 * fac4);
        let inv2 = sign_b * (vec0 * fac1 - vec1 * fac3 + vec3 * fac5);
        let inv3 = sign_a * (vec0 * fac2 - vec1 * fac4 + vec2 * fac5);

        let row0 = shuffle::<0, 0, 0, 0>(inv0, inv1);
        let row1 = shuffle::<0, 0, 0, 0>(inv2, inv3);
        let row2 = shuffle::<0, 2, 0, 2>(row0, row1);

        let det = (x * row2).element_sum();

        if det == 0.0 {
            return Self::ZERO;
        }

        let inv_det = det.recip();
        Self::from_cols(
            inv0 * inv_det,
            inv1 * inv_det,
            inv2 * inv_det,
            inv3 * inv_det,
        )
    }

    /// Returns the inverse of `self`.
    ///
    /// If the matrix is not invertible the returned matrix will be invalid.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        self.inverse_impl()
    }

    /// Returns the inverse of `self` or `Mat4A::ZERO` if the matrix is not invertible.
    #[inline]
    #[must_use]
    pub fn inverse_or_zero(&self) -> Self {
        self.inverse_impl()
    }

    /// Transforms the given 3D vector as a point, applying perspective correction.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is `1.0`.
    /// The perspective divide is performed meaning the resulting 3D vector is divided by `w`.
    ///
    /// This method assumes that `self` contains a projective transform.
    #[inline]
    #[must_use]
    pub fn project_point3(&self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.to_mat4().project_point3(rhs)
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
    pub fn transform_point3(&self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.to_mat4().transform_point3(rhs)
    }

    /// Transforms the give 3D vector as a direction.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is
    /// `0.0`.
    ///
    /// This method assumes that `self` contains a valid affine transform.
    #[inline]
    #[must_use]
    pub fn transform_vector3(&self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.to_mat4().transform_vector3(rhs)
    }

    /// Transforms a 4D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec4(&self, rhs: GVec4<f32>) -> GVec4<f32> {
        self.to_mat4().mul_vec4(rhs)
    }

    /// Transforms a SIMD-aligned 4D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec4a(&self, rhs: Vec4A) -> Vec4A {
        let mut res = self.x_axis * rhs.x;
        res += self.y_axis * rhs.y;
        res += self.z_axis * rhs.z;
        res += self.w_axis * rhs.w;
        res
    }

    /// Transforms a 4D vector by the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn mul_transpose_vec4(&self, rhs: GVec4<f32>) -> GVec4<f32> {
        self.to_mat4().mul_transpose_vec4(rhs)
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
    pub fn mul_scalar(&self, rhs: f32) -> Self {
        Self::from_cols(
            self.x_axis * rhs,
            self.y_axis * rhs,
            self.z_axis * rhs,
            self.w_axis * rhs,
        )
    }

    /// Multiply `self` by a scaling vector `scale`.
    ///
    /// This is faster than creating a whole diagonal scaling matrix and then multiplying that.
    /// This operation is commutative.
    #[inline]
    #[must_use]
    pub fn mul_diagonal_scale(&self, scale: GVec4<f32>) -> Self {
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
    pub fn div_scalar(&self, rhs: f32) -> Self {
        Self::from_cols(
            self.x_axis / rhs,
            self.y_axis / rhs,
            self.z_axis / rhs,
            self.w_axis / rhs,
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
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: f32) -> bool {
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

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite()
            & self.y_axis.is_finite()
            & self.z_axis.is_finite()
            & self.w_axis.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() | self.y_axis.is_nan() | self.z_axis.is_nan() | self.w_axis.is_nan()
    }
}

/// # Conversions
impl Mat4A {
    /// Creates a [`Mat4A`] from a [`GMat4<f32>`].
    #[inline(always)]
    #[must_use]
    pub const fn from_mat4(m: GMat4<f32>) -> Self {
        Self::from_cols(
            Vec4A::from_vec4(m.x_axis),
            Vec4A::from_vec4(m.y_axis),
            Vec4A::from_vec4(m.z_axis),
            Vec4A::from_vec4(m.w_axis),
        )
    }

    /// Converts `self` to a [`GMat4<f32>`].
    #[inline(always)]
    #[must_use]
    pub fn to_mat4(self) -> GMat4<f32> {
        GMat4::from_cols(
            self.x_axis.to_vec4(),
            self.y_axis.to_vec4(),
            self.z_axis.to_vec4(),
            self.w_axis.to_vec4(),
        )
    }

    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GMat4<U>
    where
        f32: NumCast<U>,
    {
        self.to_mat4().cast()
    }
}

impl Default for Mat4A {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl fmt::Debug for Mat4A {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Mat4A")
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis)
            .field("w_axis", &self.w_axis)
            .finish()
    }
}

impl fmt::Display for Mat4A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl Add for Mat4A {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_cols(
            self.x_axis + rhs.x_axis,
            self.y_axis + rhs.y_axis,
            self.z_axis + rhs.z_axis,
            self.w_axis + rhs.w_axis,
        )
    }
}

impl Add<&Mat4A> for Mat4A {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Mat4A) -> Self {
        self.add(*rhs)
    }
}

impl AddAssign for Mat4A {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs);
    }
}

impl Sub for Mat4A {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_cols(
            self.x_axis - rhs.x_axis,
            self.y_axis - rhs.y_axis,
            self.z_axis - rhs.z_axis,
            self.w_axis - rhs.w_axis,
        )
    }
}

impl Sub<&Mat4A> for Mat4A {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Mat4A) -> Self {
        self.sub(*rhs)
    }
}

impl SubAssign for Mat4A {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(rhs);
    }
}

impl Mul for Mat4A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_cols(
            self.mul_vec4a(rhs.x_axis),
            self.mul_vec4a(rhs.y_axis),
            self.mul_vec4a(rhs.z_axis),
            self.mul_vec4a(rhs.w_axis),
        )
    }
}

impl Mul<&Mat4A> for Mat4A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Mat4A) -> Self {
        self.mul(*rhs)
    }
}

impl MulAssign for Mat4A {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs);
    }
}

impl Mul<GVec4<f32>> for Mat4A {
    type Output = GVec4<f32>;
    #[inline]
    fn mul(self, rhs: GVec4<f32>) -> GVec4<f32> {
        self.mul_vec4(rhs)
    }
}

impl Mul<Vec4A> for Mat4A {
    type Output = Vec4A;
    #[inline]
    fn mul(self, rhs: Vec4A) -> Vec4A {
        self.mul_vec4a(rhs)
    }
}

impl Mul<f32> for Mat4A {
    type Output = Mat4A;
    #[inline]
    fn mul(self, rhs: f32) -> Mat4A {
        self.mul_scalar(rhs)
    }
}

impl Mul<Mat4A> for f32 {
    type Output = Mat4A;
    #[inline]
    fn mul(self, rhs: Mat4A) -> Mat4A {
        rhs.mul_scalar(self)
    }
}

impl MulAssign<f32> for Mat4A {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = self.mul(rhs);
    }
}

impl Div<f32> for Mat4A {
    type Output = Mat4A;
    #[inline]
    fn div(self, rhs: f32) -> Mat4A {
        self.div_scalar(rhs)
    }
}

impl DivAssign<f32> for Mat4A {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self = self.div(rhs);
    }
}

impl Neg for Mat4A {
    type Output = Mat4A;
    #[inline]
    fn neg(self) -> Mat4A {
        Self::from_cols(-self.x_axis, -self.y_axis, -self.z_axis, -self.w_axis)
    }
}

impl Sum for Mat4A {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl Product for Mat4A {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl From<GMat4<f32>> for Mat4A {
    #[inline(always)]
    fn from(m: GMat4<f32>) -> Self {
        Self::from_mat4(m)
    }
}

impl From<Mat4A> for GMat4<f32> {
    #[inline(always)]
    fn from(m: Mat4A) -> Self {
        m.to_mat4()
    }
}
