use crate::matrix::GMat2;
use crate::matrix::GMat3;
use crate::rotation::GRot2;
use crate::vector::GVec2;

use core::{
    fmt,
    iter::{Product, Sum},
    ops::*,
};

use gimd::f32x4;
use gnum::num::{NumCast, Real};

/// Creates a 2x2 column-major matrix with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Mat2`](crate::matrix::Mat2),
/// at the cost of slightly more expensive construction and field access.
#[inline(always)]
#[must_use]
pub const fn mat2a(x_axis: GVec2<f32>, y_axis: GVec2<f32>) -> Mat2A {
    Mat2A::from_cols(x_axis, y_axis)
}

/// A 2x2 column-major matrix with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Mat2`](crate::matrix::Mat2),
/// at the cost of slightly more expensive construction and field access.
///
/// # Field access
///
/// The columns live in a register rather than in struct fields, so `x_axis` and `y_axis` are reached
/// through [`Deref`].
#[derive(Clone, Copy)]
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
#[repr(transparent)]
pub struct Mat2A(f32x4);

/// # Constants
impl Mat2A {
    /// All zeros.
    pub const ZERO: Self = Self::from_cols(GVec2::ZERO, GVec2::ZERO);

    /// The 2x2 identity matrix, where all diagonal elements are `1.0` and all off-diagonal elements are `0.0`.
    pub const IDENTITY: Self = Self::from_cols(GVec2::X, GVec2::Y);

    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(GVec2::NAN, GVec2::NAN);
}

/// # Construction
impl Mat2A {
    /// Creates a 2x2 matrix from two column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: GVec2<f32>, y_axis: GVec2<f32>) -> Self {
        Self(f32x4::from_array([x_axis.x, x_axis.y, y_axis.x, y_axis.y]))
    }

    /// Creates a 2x2 matrix from a `[f32; 4]` array stored in column major order.
    /// If your data is stored in row major you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array(m: &[f32; 4]) -> Self {
        Self(f32x4::from_array([m[0], m[1], m[2], m[3]]))
    }

    /// Creates a `[f32; 4]` array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array(&self) -> [f32; 4] {
        [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    }

    /// Creates a 2x2 matrix from a `[[f32; 2]; 2]` 2D array stored in column major order.
    /// If your data is in row major order you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[f32; 2]; 2]) -> Self {
        Self::from_cols(GVec2::from_array(m[0]), GVec2::from_array(m[1]))
    }

    /// Creates a `[[f32; 2]; 2]` 2D array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub fn to_cols_array_2d(&self) -> [[f32; 2]; 2] {
        [self.x_axis.to_array(), self.y_axis.to_array()]
    }

    /// Creates a 2x2 matrix with its diagonal set to `diagonal` and all other entries set to 0.
    #[doc(alias = "scale")]
    #[inline]
    #[must_use]
    pub const fn from_diagonal(diagonal: GVec2<f32>) -> Self {
        Self::from_cols(GVec2::new(diagonal.x, 0.0), GVec2::new(0.0, diagonal.y))
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

    /// Creates a 2x2 matrix containing a non-uniform `scale` and `rotation`.
    #[inline]
    #[must_use]
    pub fn from_scale_rotation(scale: GVec2<f32>, rotation: GRot2<f32>) -> Self {
        GMat2::from_scale_rotation(scale, rotation).into()
    }

    /// Creates a 2x2 matrix containing a `rotation`.
    #[inline]
    #[must_use]
    pub fn from_rotation(rotation: GRot2<f32>) -> Self {
        GMat2::from_rotation(rotation).into()
    }

    /// Creates a 2x2 matrix from a 3x3 matrix, discarding the 3rd row and column.
    #[inline]
    #[must_use]
    pub fn from_mat3(m: GMat3<f32>) -> Self {
        GMat2::from_mat3(m).into()
    }

    /// Creates a 2x2 matrix from the minor of the given 3x3 matrix, discarding the `i`th column
    /// and `j`th row.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is greater than 2.
    #[inline]
    #[must_use]
    pub fn from_mat3_minor(m: GMat3<f32>, i: usize, j: usize) -> Self {
        GMat2::from_mat3_minor(m, i, j).into()
    }

    /// Creates a 2x2 matrix from the first 4 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice(slice: &[f32]) -> Self {
        Self(f32x4::from_array([slice[0], slice[1], slice[2], slice[3]]))
    }

    /// Writes the columns of `self` to the first 4 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [f32]) {
        slice[0] = self.x_axis.x;
        slice[1] = self.x_axis.y;
        slice[2] = self.y_axis.x;
        slice[3] = self.y_axis.y;
    }
}

/// # Operations
impl Mat2A {
    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 1.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> GVec2<f32> {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns a mutable reference to the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 1.
    #[inline]
    pub fn col_mut(&mut self, index: usize) -> &mut GVec2<f32> {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the matrix row for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 1.
    #[inline]
    #[must_use]
    pub fn row(&self, index: usize) -> GVec2<f32> {
        match index {
            0 => GVec2::new(self.x_axis.x, self.y_axis.x),
            1 => GVec2::new(self.x_axis.y, self.y_axis.y),
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Self {
        self.to_mat2().transpose().into()
    }

    /// Returns the diagonal of `self`.
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> GVec2<f32> {
        GVec2::new(self.x_axis.x, self.y_axis.y)
    }

    /// Returns the determinant of `self`.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> f32 {
        self.to_mat2().determinant()
    }

    /// Returns the inverse of `self`.
    ///
    /// If the matrix is not invertible the returned matrix will be invalid.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        self.to_mat2().inverse().into()
    }

    /// Returns the inverse of `self` or `Mat2A::ZERO` if the matrix is not invertible.
    #[inline]
    #[must_use]
    pub fn inverse_or_zero(&self) -> Self {
        self.to_mat2().inverse_or_zero().into()
    }

    /// Transforms a 2D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec2(&self, rhs: GVec2<f32>) -> GVec2<f32> {
        self.to_mat2().mul_vec2(rhs)
    }

    /// Transforms a 2D vector by the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn mul_transpose_vec2(&self, rhs: GVec2<f32>) -> GVec2<f32> {
        self.to_mat2().mul_transpose_vec2(rhs)
    }

    /// Multiplies two 2x2 matrices.
    #[inline]
    #[must_use]
    pub fn mul_mat2(&self, rhs: &Self) -> Self {
        self.mul(rhs)
    }

    /// Adds two 2x2 matrices.
    #[inline]
    #[must_use]
    pub fn add_mat2(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }

    /// Subtracts two 2x2 matrices.
    #[inline]
    #[must_use]
    pub fn sub_mat2(&self, rhs: &Self) -> Self {
        self.sub(rhs)
    }

    /// Multiplies a 2x2 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn mul_scalar(&self, rhs: f32) -> Self {
        Self(self.0 * f32x4::splat(rhs))
    }

    /// Multiply `self` by a scaling vector `scale`.
    /// This is faster than creating a whole diagonal scaling matrix and then multiplying that.
    /// This operation is commutative.
    #[inline]
    #[must_use]
    pub fn mul_diagonal_scale(&self, scale: GVec2<f32>) -> Self {
        Self::from_cols(self.x_axis * scale.x, self.y_axis * scale.y)
    }

    /// Divides a 2x2 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn div_scalar(&self, rhs: f32) -> Self {
        Self(self.0 / f32x4::splat(rhs))
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
        self.to_mat2().abs_diff_eq(rhs.to_mat2(), max_abs_diff)
    }

    /// Takes the absolute value of each element in `self`.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self(gnum::num::Signed::abs(self.0))
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite() & self.y_axis.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() | self.y_axis.is_nan()
    }
}

/// # Conversions
impl Mat2A {
    /// Creates a [`Mat2A`] from a [`GMat2<f32>`].
    #[inline(always)]
    #[must_use]
    pub const fn from_mat2(m: GMat2<f32>) -> Self {
        Self::from_cols(m.x_axis, m.y_axis)
    }

    /// Converts `self` to a [`GMat2<f32>`].
    #[inline(always)]
    #[must_use]
    pub fn to_mat2(self) -> GMat2<f32> {
        GMat2::from_cols(self.x_axis, self.y_axis)
    }

    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GMat2<U>
    where
        f32: NumCast<U>,
    {
        self.to_mat2().cast()
    }
}

impl Deref for Mat2A {
    type Target = crate::deref::Cols2<GVec2<f32>>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl DerefMut for Mat2A {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl PartialEq for Mat2A {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.x_axis == rhs.x_axis && self.y_axis == rhs.y_axis
    }
}

impl Default for Mat2A {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl fmt::Debug for Mat2A {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Mat2A")
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .finish()
    }
}

impl fmt::Display for Mat2A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(p) = f.precision() {
            write!(f, "[{:.*}, {:.*}]", p, self.x_axis, p, self.y_axis)
        } else {
            write!(f, "[{}, {}]", self.x_axis, self.y_axis)
        }
    }
}

impl Add for Mat2A {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Add<&Mat2A> for Mat2A {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Mat2A) -> Self {
        self.add(*rhs)
    }
}

impl AddAssign for Mat2A {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs);
    }
}

impl Sub for Mat2A {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Sub<&Mat2A> for Mat2A {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Mat2A) -> Self {
        self.sub(*rhs)
    }
}

impl SubAssign for Mat2A {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(rhs);
    }
}

impl Mul for Mat2A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        (self.to_mat2() * rhs.to_mat2()).into()
    }
}

impl Mul<&Mat2A> for Mat2A {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Mat2A) -> Self {
        self.mul(*rhs)
    }
}

impl MulAssign for Mat2A {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs);
    }
}

impl Mul<GVec2<f32>> for Mat2A {
    type Output = GVec2<f32>;
    #[inline]
    fn mul(self, rhs: GVec2<f32>) -> GVec2<f32> {
        self.mul_vec2(rhs)
    }
}

impl Mul<f32> for Mat2A {
    type Output = Mat2A;
    #[inline]
    fn mul(self, rhs: f32) -> Mat2A {
        self.mul_scalar(rhs)
    }
}

impl Mul<Mat2A> for f32 {
    type Output = Mat2A;
    #[inline]
    fn mul(self, rhs: Mat2A) -> Mat2A {
        rhs.mul_scalar(self)
    }
}

impl MulAssign<f32> for Mat2A {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = self.mul(rhs);
    }
}

impl Div<f32> for Mat2A {
    type Output = Mat2A;
    #[inline]
    fn div(self, rhs: f32) -> Mat2A {
        self.div_scalar(rhs)
    }
}

impl DivAssign<f32> for Mat2A {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self = self.div(rhs);
    }
}

impl Neg for Mat2A {
    type Output = Mat2A;
    #[inline]
    fn neg(self) -> Mat2A {
        Self(-self.0)
    }
}

impl Sum for Mat2A {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl Product for Mat2A {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl From<GMat2<f32>> for Mat2A {
    #[inline(always)]
    fn from(m: GMat2<f32>) -> Self {
        Self::from_mat2(m)
    }
}

impl From<Mat2A> for GMat2<f32> {
    #[inline(always)]
    fn from(m: Mat2A) -> Self {
        m.to_mat2()
    }
}
