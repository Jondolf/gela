use crate::matrix::GMat3;
use crate::rotation::GRot2;
use crate::vector::{GVec2, Vec3Swizzles};

use core::{
    iter::{Product, Sum},
    ops::*,
};

use gnum::{
    num::{Float, NumCast, Real},
    simd::{MaskLike, Select},
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// Creates a 2x2 matrix from two column vectors.
#[inline(always)]
#[must_use]
pub const fn gmat2<T: Real>(x_axis: GVec2<T>, y_axis: GVec2<T>) -> GMat2<T> {
    GMat2::from_cols(x_axis, y_axis)
}

/// A 2x2 column major matrix.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[cfg_attr(
    feature = "zerocopy",
    derive(FromBytes, Immutable, IntoBytes, KnownLayout)
)]
#[repr(align(16))]
#[cfg_attr(feature = "cuda", repr(align(8)))]
#[repr(C)]
pub struct GMat2<T: Real> {
    /// The first column of the matrix.
    pub x_axis: GVec2<T>,
    /// The second column of the matrix.
    pub y_axis: GVec2<T>,
}

/// # Constants
impl<T: Real> GMat2<T> {
    /// All zeros.
    pub const ZERO: Self = Self::from_cols(GVec2::ZERO, GVec2::ZERO);

    /// The 2x2 identity matrix, where all diagonal elements are `1.0` and all off-diagonal elements are `0.0`.
    pub const IDENTITY: Self = Self::from_cols(GVec2::X, GVec2::Y);
}

impl<T: Float> GMat2<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::from_cols(GVec2::NAN, GVec2::NAN);
}

/// # Construction
impl<T: Real> GMat2<T> {
    #[inline(always)]
    #[must_use]
    const fn new(m00: T, m01: T, m10: T, m11: T) -> Self {
        Self {
            x_axis: GVec2::new(m00, m01),
            y_axis: GVec2::new(m10, m11),
        }
    }

    /// Creates a 2x2 matrix from two column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: GVec2<T>, y_axis: GVec2<T>) -> Self {
        Self { x_axis, y_axis }
    }

    /// Creates a 2x2 matrix from a `[T; 4]` array stored in column major order.
    /// If your data is stored in row major you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array(m: &[T; 4]) -> Self {
        Self::new(m[0], m[1], m[2], m[3])
    }

    /// Creates a `[T; 4]` array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array(&self) -> [T; 4] {
        [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    }

    /// Creates a 2x2 matrix from a `[[T; 2]; 2]` 2D array stored in column major order.
    /// If your data is in row major order you will need to `transpose` the returned
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[T; 2]; 2]) -> Self {
        Self::from_cols(GVec2::from_array(m[0]), GVec2::from_array(m[1]))
    }

    /// Creates a `[[T; 2]; 2]` 2D array storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[T; 2]; 2] {
        [self.x_axis.to_array(), self.y_axis.to_array()]
    }

    /// Creates a 2x2 matrix with its diagonal set to `diagonal` and all other entries set to 0.
    #[doc(alias = "scale")]
    #[inline]
    #[must_use]
    pub const fn from_diagonal(diagonal: GVec2<T>) -> Self {
        Self::new(diagonal.x, T::ZERO, T::ZERO, diagonal.y)
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
            x_axis: GVec2::select(boolean, if_true.x_axis, if_false.x_axis),
            y_axis: GVec2::select(boolean, if_true.y_axis, if_false.y_axis),
        }
    }

    /// Creates a 2x2 matrix containing a non-uniform `scale` and `rotation`.
    #[inline]
    #[must_use]
    pub fn from_scale_rotation(scale: GVec2<T>, rotation: GRot2<T>) -> Self {
        let (sin, cos) = (rotation.sin, rotation.cos);
        Self::new(cos * scale.x, sin * scale.x, -sin * scale.y, cos * scale.y)
    }

    /// Creates a 2x2 matrix containing a `rotation`.
    #[inline]
    #[must_use]
    pub fn from_rotation(rotation: GRot2<T>) -> Self {
        let (sin, cos) = (rotation.sin, rotation.cos);
        Self::new(cos, sin, -sin, cos)
    }

    /// Creates a 2x2 matrix from a 3x3 matrix, discarding the 3rd row and column.
    #[inline]
    #[must_use]
    pub fn from_mat3(m: GMat3<T>) -> Self {
        Self::from_cols(m.x_axis.xy(), m.y_axis.xy())
    }

    /// Creates a 2x2 matrix from the minor of the given 3x3 matrix, discarding the `i`th column
    /// and `j`th row.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is greater than 2.
    #[inline]
    #[must_use]
    pub fn from_mat3_minor(m: GMat3<T>, i: usize, j: usize) -> Self {
        match (i, j) {
            (0, 0) => Self::from_cols(m.y_axis.yz(), m.z_axis.yz()),
            (0, 1) => Self::from_cols(m.y_axis.xz(), m.z_axis.xz()),
            (0, 2) => Self::from_cols(m.y_axis.xy(), m.z_axis.xy()),
            (1, 0) => Self::from_cols(m.x_axis.yz(), m.z_axis.yz()),
            (1, 1) => Self::from_cols(m.x_axis.xz(), m.z_axis.xz()),
            (1, 2) => Self::from_cols(m.x_axis.xy(), m.z_axis.xy()),
            (2, 0) => Self::from_cols(m.x_axis.yz(), m.y_axis.yz()),
            (2, 1) => Self::from_cols(m.x_axis.xz(), m.y_axis.xz()),
            (2, 2) => Self::from_cols(m.x_axis.xy(), m.y_axis.xy()),
            _ => panic!("index out of bounds"),
        }
    }

    /// Creates a 2x2 matrix from the first 4 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice(slice: &[T]) -> Self {
        Self::new(slice[0], slice[1], slice[2], slice[3])
    }

    /// Writes the columns of `self` to the first 4 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [T]) {
        slice[0] = self.x_axis.x;
        slice[1] = self.x_axis.y;
        slice[2] = self.y_axis.x;
        slice[3] = self.y_axis.y;
    }
}

/// # Operations
impl<T: Real> GMat2<T> {
    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 1.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> GVec2<T> {
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
    pub fn col_mut(&mut self, index: usize) -> &mut GVec2<T> {
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
    pub fn row(&self, index: usize) -> GVec2<T> {
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
        Self {
            x_axis: GVec2::new(self.x_axis.x, self.y_axis.x),
            y_axis: GVec2::new(self.x_axis.y, self.y_axis.y),
        }
    }

    /// Returns the diagonal of `self`.
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> GVec2<T> {
        GVec2::new(self.x_axis.x, self.y_axis.y)
    }

    /// Returns the determinant of `self`.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> T {
        self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
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
        let det = self.determinant();
        let inv_det = det.recip();
        let inverted = Self::new(
            self.y_axis.y * inv_det,
            self.x_axis.y * -inv_det,
            self.y_axis.x * -inv_det,
            self.x_axis.x * inv_det,
        );
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

    /// Transforms a 2D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec2(&self, rhs: GVec2<T>) -> GVec2<T> {
        #[allow(clippy::suspicious_operation_groupings)]
        GVec2::new(
            (self.x_axis.x * rhs.x) + (self.y_axis.x * rhs.y),
            (self.x_axis.y * rhs.x) + (self.y_axis.y * rhs.y),
        )
    }

    /// Transforms a 2D vector by the transpose of `self`.
    #[inline]
    #[must_use]
    pub fn mul_transpose_vec2(&self, rhs: GVec2<T>) -> GVec2<T> {
        GVec2::new(self.x_axis.dot(rhs), self.y_axis.dot(rhs))
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
    pub fn mul_scalar(&self, rhs: T) -> Self {
        Self::from_cols(self.x_axis.mul(rhs), self.y_axis.mul(rhs))
    }

    /// Multiply `self` by a scaling vector `scale`.
    /// This is faster than creating a whole diagonal scaling matrix and then multiplying that.
    /// This operation is commutative.
    #[inline]
    #[must_use]
    pub fn mul_diagonal_scale(&self, scale: GVec2<T>) -> Self {
        Self::from_cols(self.x_axis * scale.x, self.y_axis * scale.y)
    }

    /// Divides a 2x2 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn div_scalar(&self, rhs: T) -> Self {
        let rhs = GVec2::splat(rhs);
        Self::from_cols(self.x_axis.div(rhs), self.y_axis.div(rhs))
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
    }

    /// Takes the absolute value of each element in `self`.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self::from_cols(self.x_axis.abs(), self.y_axis.abs())
    }
}

impl<T: Float> GMat2<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> T::Bool {
        self.x_axis.is_finite() & self.y_axis.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> T::Bool {
        self.x_axis.is_nan() | self.y_axis.is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> GMat2<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar matrix into a SIMD matrix, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: GMat2<T::Element>) -> Self {
        Self::from_cols(
            GVec2::broadcast(value.x_axis),
            GVec2::broadcast(value.y_axis),
        )
    }

    /// Extracts the i-th lane of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub fn extract(&self, i: usize) -> GMat2<T::Element> {
        GMat2::from_cols(self.x_axis.extract(i), self.y_axis.extract(i))
    }

    /// Extracts the i-th lane of `self` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub unsafe fn extract_unchecked(&self, i: usize) -> GMat2<T::Element> {
        unsafe {
            GMat2::from_cols(
                self.x_axis.extract_unchecked(i),
                self.y_axis.extract_unchecked(i),
            )
        }
    }

    /// Replaces the i-th lane of `self` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    pub fn replace(&mut self, i: usize, value: GMat2<T::Element>) {
        self.x_axis.replace(i, value.x_axis);
        self.y_axis.replace(i, value.y_axis);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: GMat2<T::Element>) {
        unsafe {
            self.x_axis.replace_unchecked(i, value.x_axis);
            self.y_axis.replace_unchecked(i, value.y_axis);
        }
    }
}

/// # Conversion
impl<T: Real> GMat2<T> {
    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GMat2<U>
    where
        T: NumCast<U>,
    {
        GMat2::from_cols(self.x_axis.cast(), self.y_axis.cast())
    }
}

impl<T: Real> Default for GMat2<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real + Add<Output = T>> Add for GMat2<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: GMat2<T>) -> Self {
        GMat2::from_cols(self.x_axis.add(rhs.x_axis), self.y_axis.add(rhs.y_axis))
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat2<T>> for GMat2<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &GMat2<T>) -> Self {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<GMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: GMat2<T>) -> GMat2<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: &GMat2<T>) -> GMat2<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> AddAssign for GMat2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: GMat2<T>) {
        *self = self.add(rhs);
    }
}

impl<T: Real + AddAssign> AddAssign<&GMat2<T>> for GMat2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &GMat2<T>) {
        self.add_assign(*rhs);
    }
}

impl<T: Real + Sub<Output = T>> Sub for GMat2<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: GMat2<T>) -> Self {
        GMat2::from_cols(self.x_axis.sub(rhs.x_axis), self.y_axis.sub(rhs.y_axis))
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat2<T>> for GMat2<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &GMat2<T>) -> Self {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<GMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: GMat2<T>) -> GMat2<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: &GMat2<T>) -> GMat2<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> SubAssign for GMat2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: GMat2<T>) {
        *self = self.sub(rhs);
    }
}

impl<T: Real + SubAssign> SubAssign<&GMat2<T>> for GMat2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &GMat2<T>) {
        self.sub_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul for GMat2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: GMat2<T>) -> Self {
        GMat2::from_cols(self.mul_vec2(rhs.x_axis), self.mul_vec2(rhs.y_axis))
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat2<T>> for GMat2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &GMat2<T>) -> Self {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: GMat2<T>) -> GMat2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: &GMat2<T>) -> GMat2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign for GMat2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GMat2<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + MulAssign> MulAssign<&GMat2<T>> for GMat2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GMat2<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec2<T>> for GMat2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: GVec2<T>) -> GVec2<T> {
        self.mul_vec2(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec2<T>> for GMat2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: &GVec2<T>) -> GVec2<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec2<T>> for &GMat2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec2<T>> for &GMat2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: T) -> GMat2<T> {
        self.mul_scalar(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GMat2<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: T) -> GMat2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GMat2<T> {
        (*self).mul(*rhs)
    }
}

// We cannot implement scalar * matrix generically because of Rust's orphan rules.
macro_rules! impl_scalar_left_mul {
    ($($t:ty),*) => {
        $(
            impl Mul<GMat2<$t>> for $t {
                type Output = GMat2<$t>;
                #[inline]
                fn mul(self, rhs: GMat2<$t>) -> GMat2<$t> {
                    rhs.mul_scalar(self)
                }
            }

            impl Mul<&GMat2<$t>> for $t {
                type Output = GMat2<$t>;
                #[inline]
                fn mul(self, rhs: &GMat2<$t>) -> GMat2<$t> {
                    self.mul(*rhs)
                }
            }

            impl Mul<GMat2<$t>> for &$t {
                type Output = GMat2<$t>;
                #[inline]
                fn mul(self, rhs: GMat2<$t>) -> GMat2<$t> {
                    (*self).mul(rhs)
                }
            }

            impl Mul<&GMat2<$t>> for &$t {
                type Output = GMat2<$t>;
                #[inline]
                fn mul(self, rhs: &GMat2<$t>) -> GMat2<$t> {
                    (*self).mul(*rhs)
                }
            }
        )*
    };
}

// TODO: Implement for SIMD types
impl_scalar_left_mul!(f32, f64);

impl<T: Real + Mul<Output = T>> MulAssign<T> for GMat2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&T> for GMat2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Div<Output = T>> Div<T> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn div(self, rhs: T) -> GMat2<T> {
        self.div_scalar(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn div(self, rhs: &T) -> GMat2<T> {
        self.div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn div(self, rhs: T) -> GMat2<T> {
        (*self).div(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn div(self, rhs: &T) -> GMat2<T> {
        (*self).div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> DivAssign<T> for GMat2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = self.div(rhs);
    }
}

impl<T: Real + Div<Output = T>> DivAssign<&T> for GMat2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.div_assign(*rhs);
    }
}

impl<T: Real + Neg<Output = T>> Neg for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn neg(self) -> GMat2<T> {
        GMat2::from_cols(self.x_axis.neg(), self.y_axis.neg())
    }
}

impl<T: Real + Neg<Output = T>> Neg for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn neg(self) -> GMat2<T> {
        (*self).neg()
    }
}

impl<T: Real> Sum<GMat2<T>> for GMat2<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, T: Real> Sum<&'a GMat2<T>> for GMat2<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl<T: Real> Product<GMat2<T>> for GMat2<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl<'a, T: Real> Product<&'a GMat2<T>> for GMat2<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| a * b)
    }
}

impl<T: Real> AsRef<[T; 4]> for GMat2<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 4] {
        unsafe { &*(self as *const GMat2<T> as *const [T; 4]) }
    }
}

impl<T: Real> AsMut<[T; 4]> for GMat2<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 4] {
        unsafe { &mut *(self as *mut GMat2<T> as *mut [T; 4]) }
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for GMat2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(p) = f.precision() {
            write!(f, "[{:.*}, {:.*}]", p, self.x_axis, p, self.y_axis)
        } else {
            write!(f, "[{}, {}]", self.x_axis, self.y_axis)
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for GMat2<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(Mat2))
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .finish()
    }
}
