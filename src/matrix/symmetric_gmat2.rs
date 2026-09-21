use crate::matrix::GMat2;
#[cfg(feature = "simd")]
use crate::matrix::Mat2A;
use crate::vector::GVec2;

use core::{iter::Sum, ops::*};

use gnum::{
    num::{Float, NumCast, Real},
    simd::{MaskLike, Select},
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// Creates a symmetric 2x2 matrix from its bottom left triangle, including diagonal elements.
///
/// The elements are in column-major order `mCR`, where `C` is the column index
/// and `R` is the row index.
#[inline(always)]
#[must_use]
pub const fn symmetric_gmat2<T: Real>(m00: T, m01: T, m11: T) -> SymmetricGMat2<T> {
    SymmetricGMat2::new(m00, m01, m11)
}

/// The bottom left triangle (including the diagonal) of a symmetric 2x2 column-major matrix.
///
/// This is useful for storing a symmetric 2x2 matrix in a more compact form and performing some
/// matrix operations more efficiently.
///
/// Defining properties of symmetric matrices include:
///
/// - The matrix is equal to its transpose.
/// - The matrix has real eigenvalues.
/// - The eigenvectors corresponding to the eigenvalues are orthogonal.
/// - The matrix is always diagonalizable.
///
/// The sum and difference of two symmetric matrices is always symmetric.
/// However, the product of two symmetric matrices is *only* symmetric
/// if the matrices are commutable, meaning that `AB == BA`.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "zerocopy", derive(FromBytes, Immutable, KnownLayout))]
#[repr(C)]
pub struct SymmetricGMat2<T: Real> {
    /// The first element of the first column.
    pub m00: T,
    /// The second element of the first column.
    pub m01: T,
    /// The second element of the second column.
    pub m11: T,
}

/// # Constants
impl<T: Real> SymmetricGMat2<T> {
    /// All zeros.
    pub const ZERO: Self = Self::new(T::ZERO, T::ZERO, T::ZERO);

    /// The 2x2 identity matrix, where all diagonal elements are `1.0` and all off-diagonal elements are `0.0`.
    pub const IDENTITY: Self = Self::new(T::ONE, T::ZERO, T::ONE);
}

impl<T: Float> SymmetricGMat2<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::new(T::NAN, T::NAN, T::NAN);
}

/// # Constructors
impl<T: Real> SymmetricGMat2<T> {
    /// Creates a new symmetric 2x2 matrix from its bottom left triangle, including diagonal elements.
    ///
    /// The elements are in column-major order `mCR`, where `C` is the column index
    /// and `R` is the row index.
    #[inline(always)]
    #[must_use]
    pub const fn new(m00: T, m01: T, m11: T) -> Self {
        Self { m00, m01, m11 }
    }

    /// Creates a symmetric 2x2 matrix from a `[T; 3]` containing the lower left triangle
    /// of the matrix in column-major order `[m00, m01, m11]`.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(m: [T; 3]) -> Self {
        Self::new(m[0], m[1], m[2])
    }

    /// Creates a `[T; 3]` array containing the lower left triangle of the matrix
    /// in column-major order `[m00, m01, m11]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(&self) -> [T; 3] {
        [self.m00, self.m01, self.m11]
    }

    /// Creates a symmetric 2x2 matrix from the lower left triangle of two column vectors.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_upper()`] for the upper triangular version.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_lower(x_axis: GVec2<T>, y_axis: GVec2<T>) -> Self {
        Self::new(x_axis.x, x_axis.y, y_axis.y)
    }

    /// Creates a symmetric 2x2 matrix from the upper right triangle of two column vectors.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_lower()`] for the lower triangular version.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_upper(x_axis: GVec2<T>, y_axis: GVec2<T>) -> Self {
        Self::new(x_axis.x, y_axis.x, y_axis.y)
    }

    /// Creates a symmetric 2x2 matrix from the lower left triangle of a `[T; 4]` array
    /// stored in column major order.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_array_upper()`] for the upper triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_lower(m: &[T; 4]) -> Self {
        Self::new(m[0], m[1], m[3])
    }

    /// Creates a symmetric 2x2 matrix from the upper right triangle of a `[T; 4]` array
    /// stored in column major order.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_array_lower()`] for the lower triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_upper(m: &[T; 4]) -> Self {
        Self::new(m[0], m[2], m[3])
    }

    /// Creates a `[T; 4]` array storing data in column major order.
    #[inline]
    #[must_use]
    pub const fn to_cols_array(&self) -> [T; 4] {
        [self.m00, self.m01, self.m01, self.m11]
    }

    /// Creates a symmetric 2x2 matrix from the lower left triangle of a `[[T; 2]; 2]` 2D array
    /// stored in column major order.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_array_2d_upper()`] for the upper triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d_lower(m: &[[T; 2]; 2]) -> Self {
        Self::from_cols_lower(GVec2::from_array(m[0]), GVec2::from_array(m[1]))
    }

    /// Creates a symmetric 2x2 matrix from the upper right triangle of a `[[T; 2]; 2]` 2D array
    /// stored in column major order.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_array_2d_lower()`] for the lower triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d_upper(m: &[[T; 2]; 2]) -> Self {
        Self::from_cols_upper(GVec2::from_array(m[0]), GVec2::from_array(m[1]))
    }

    /// Creates a `[[T; 2]; 2]` 2D array storing data in column major order.
    #[inline]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[T; 2]; 2] {
        [[self.m00, self.m01], [self.m01, self.m11]]
    }

    /// Creates a symmetric 2x2 matrix with its diagonal set to `diagonal` and all other entries set to 0.
    #[doc(alias = "scale")]
    #[inline]
    #[must_use]
    pub const fn from_diagonal(diagonal: GVec2<T>) -> Self {
        Self::new(diagonal.x, T::ZERO, diagonal.y)
    }

    /// Creates a symmetric 2x2 matrix from the elements in `if_true` and `if_false`
    /// selecting which to use based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select<B: Select<T>>(boolean: B, if_true: Self, if_false: Self) -> Self {
        Self::new(
            B::select(boolean, if_true.m00, if_false.m00),
            B::select(boolean, if_true.m01, if_false.m01),
            B::select(boolean, if_true.m11, if_false.m11),
        )
    }

    /// Creates a symmetric 2x2 matrix from the lower left triangle of a 2x2 matrix.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_mat2_upper()`] for the upper triangular version.
    #[inline]
    #[must_use]
    pub const fn from_mat2_lower(mat: GMat2<T>) -> Self {
        Self::from_cols_lower(mat.x_axis, mat.y_axis)
    }

    /// Creates a symmetric 2x2 matrix from the upper right triangle of a 2x2 matrix.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_mat2_lower()`] for the lower triangular version.
    #[inline]
    #[must_use]
    pub const fn from_mat2_upper(mat: GMat2<T>) -> Self {
        Self::from_cols_upper(mat.x_axis, mat.y_axis)
    }

    /// Creates a symmetric 2x2 matrix from the symmetric part `(M + Mᵀ) / 2` of a 2x2 matrix.
    #[inline]
    #[must_use]
    pub fn from_mat2_symmetric_part(mat: GMat2<T>) -> Self {
        Self::new(
            mat.x_axis.x,
            (mat.x_axis.y + mat.y_axis.x) * T::HALF,
            mat.y_axis.y,
        )
    }

    /// Creates a symmetric 2x2 matrix from the product `M * Mᵀ` of a 2x2 matrix.
    ///
    /// This is the Gram matrix of the rows of `mat`. For the Gram matrix of the columns,
    /// see [`Self::from_mat2_transpose_mul()`].
    ///
    /// The result is always symmetric and positive semi-definite.
    #[inline]
    #[must_use]
    pub fn from_mat2_mul_transpose(mat: GMat2<T>) -> Self {
        let (r0, r1) = (mat.row(0), mat.row(1));
        Self::new(r0.dot(r0), r1.dot(r0), r1.dot(r1))
    }

    /// Creates a symmetric 2x2 matrix from the product `Mᵀ * M` of a 2x2 matrix.
    ///
    /// This is the Gram matrix of the columns of `mat`. For the Gram matrix of the rows,
    /// see [`Self::from_mat2_mul_transpose()`].
    ///
    /// The result is always symmetric and positive semi-definite.
    #[inline]
    #[must_use]
    pub fn from_mat2_transpose_mul(mat: GMat2<T>) -> Self {
        let (c0, c1) = (mat.x_axis, mat.y_axis);
        Self::new(c0.dot(c0), c1.dot(c0), c1.dot(c1))
    }

    /// Creates a 2x2 matrix from the symmetric 2x2 matrix in `self`.
    #[inline]
    #[must_use]
    pub const fn to_mat2(&self) -> GMat2<T> {
        GMat2::from_cols_array(&self.to_cols_array())
    }

    /// Creates a new symmetric 2x2 matrix from the outer product `v * vᵀ`.
    #[inline(always)]
    #[must_use]
    pub fn from_outer_product(v: GVec2<T>) -> Self {
        Self::new(v.x * v.x, v.x * v.y, v.y * v.y)
    }

    /// Creates a symmetric 2x2 matrix from a slice containing the lower left triangle
    /// of the matrix in column-major order `[m00, m01, m11]`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 3 elements long.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[T]) -> Self {
        Self::new(slice[0], slice[1], slice[2])
    }

    /// Writes the lower left triangle of `self` to a slice in column-major order
    /// `[m00, m01, m11]`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 3 elements long.
    #[inline]
    pub fn write_to_slice(&self, slice: &mut [T]) {
        slice[0] = self.m00;
        slice[1] = self.m01;
        slice[2] = self.m11;
    }

    /// Creates a symmetric 2x2 matrix from the lower left triangle of the first 4 values
    /// in `slice`, stored in column major order.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_slice_upper()`] for the upper triangular version.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice_lower(slice: &[T]) -> Self {
        Self::new(slice[0], slice[1], slice[3])
    }

    /// Creates a symmetric 2x2 matrix from the upper right triangle of the first 4 values
    /// in `slice`, stored in column major order.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_slice_lower()`] for the lower triangular version.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice_upper(slice: &[T]) -> Self {
        Self::new(slice[0], slice[2], slice[3])
    }

    /// Writes the columns of `self` to the first 4 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 4 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [T]) {
        slice[0] = self.m00;
        slice[1] = self.m01;
        slice[2] = self.m01;
        slice[3] = self.m11;
    }
}

/// # Operations
impl<T: Real> SymmetricGMat2<T> {
    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 1.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> GVec2<T> {
        match index {
            0 => GVec2::new(self.m00, self.m01),
            1 => GVec2::new(self.m01, self.m11),
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
        self.col(index)
    }

    /// Returns the diagonal of `self`.
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> GVec2<T> {
        GVec2::new(self.m00, self.m11)
    }

    /// Returns the trace of `self`, the sum of the diagonal elements.
    ///
    /// This is also the sum of the eigenvalues of `self`.
    #[inline]
    #[must_use]
    pub fn trace(&self) -> T {
        self.m00 + self.m11
    }

    /// Returns the adjugate of `self`, the transpose of its cofactor matrix.
    ///
    /// The adjugate of a symmetric matrix is symmetric. It satisfies
    /// `self * self.adjugate() == SymmetricGMat2::IDENTITY * self.determinant()`,
    /// and is the numerator of the inverse.
    #[inline]
    #[must_use]
    pub fn adjugate(&self) -> Self {
        // A = [ a c ]
        //     | c b |
        let [a, b] = [self.m00, self.m11];
        let c = self.m01;
        Self::new(b, -c, a)
    }

    #[inline(always)]
    #[must_use]
    fn adjugate_and_determinant(&self) -> (Self, T) {
        let adjugate = self.adjugate();
        let determinant = adjugate.m00 * self.m00 + adjugate.m01 * self.m01;
        (adjugate, determinant)
    }

    /// Returns the determinant of `self`.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> T {
        self.adjugate_and_determinant().1
    }

    /// Returns the inverse of `self`.
    ///
    /// If the matrix is not invertible the returned matrix will be invalid.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let (adjugate, determinant) = self.adjugate_and_determinant();
        adjugate.mul_scalar(determinant.recip())
    }

    /// Returns the inverse of `self` or `SymmetricGMat2::ZERO` if the matrix is not invertible.
    #[inline]
    #[must_use]
    pub fn inverse_or_zero(&self) -> Self {
        let (adjugate, determinant) = self.adjugate_and_determinant();
        let non_invertible = determinant.num_eq(T::ZERO);

        if non_invertible.all() {
            return Self::ZERO;
        }

        let inverted = adjugate.mul_scalar(determinant.recip());

        Self::select(non_invertible, Self::ZERO, inverted)
    }

    /// Transforms a 2D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec2(&self, rhs: GVec2<T>) -> GVec2<T> {
        let mut res = self.col(0).mul(rhs.x);
        res = res.add(self.col(1).mul(rhs.y));
        res
    }

    /// Multiplies two symmetric 2x2 matrices.
    ///
    /// The product of two symmetric matrices is *only* symmetric if the matrices commute,
    /// meaning that `AB == BA`. This is not the case in general, so the result is returned
    /// as a [`GMat2`].
    ///
    /// For a product that is guaranteed to be symmetric, see [`Self::congruence()`].
    #[inline]
    #[must_use]
    pub fn mul_symmetric_mat2(&self, rhs: &Self) -> GMat2<T> {
        GMat2::from_cols(self.mul_vec2(rhs.col(0)), self.mul_vec2(rhs.col(1)))
    }

    /// Multiplies a symmetric 2x2 matrix by a 2x2 matrix.
    ///
    /// The result is not symmetric in general, so it is returned as a [`GMat2`].
    #[inline]
    #[must_use]
    pub fn mul_mat2(&self, rhs: &GMat2<T>) -> GMat2<T> {
        GMat2::from_cols(self.mul_vec2(rhs.x_axis), self.mul_vec2(rhs.y_axis))
    }

    /// Returns the congruence transform `m * self * mᵀ`.
    ///
    /// The result is guaranteed to be symmetric, even if `m` is not.
    ///
    /// If `m` is a rotation matrix, `mᵀ == m⁻¹`, and this re-expresses `self` in a rotated basis.
    #[inline]
    #[must_use]
    pub fn congruence(&self, m: &GMat2<T>) -> Self {
        let (r0, r1) = (m.row(0), m.row(1));
        let (t0, t1) = (self.mul_vec2(r0), self.mul_vec2(r1));
        Self::new(r0.dot(t0), r1.dot(t0), r1.dot(t1))
    }

    /// Evaluates the quadratic form of `self` at `v`, returning `vᵀ * self * v`.
    ///
    /// In physics simulations, for an inverse mass matrix,
    /// this is the effective mass along a single axis.
    #[doc(alias = "vector_sandwich")]
    #[inline]
    #[must_use]
    pub fn quadratic_form(&self, v: GVec2<T>) -> T {
        v.dot(self.mul_vec2(v))
    }

    /// Adds two symmetric 2x2 matrices.
    #[inline]
    #[must_use]
    pub fn add_symmetric_mat2(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }

    /// Subtracts two symmetric 2x2 matrices.
    #[inline]
    #[must_use]
    pub fn sub_symmetric_mat2(&self, rhs: &Self) -> Self {
        self.sub(rhs)
    }

    /// Adds a symmetric 2x2 matrix and a 2x2 matrix.
    ///
    /// The result is not symmetric in general, so it is returned as a [`GMat2`].
    #[inline]
    #[must_use]
    pub fn add_mat2(&self, rhs: &GMat2<T>) -> GMat2<T> {
        self.to_mat2().add(rhs)
    }

    /// Subtracts a 2x2 matrix from a symmetric 2x2 matrix.
    ///
    /// The result is not symmetric in general, so it is returned as a [`GMat2`].
    #[inline]
    #[must_use]
    pub fn sub_mat2(&self, rhs: &GMat2<T>) -> GMat2<T> {
        self.to_mat2().sub(rhs)
    }

    /// Multiplies a symmetric 2x2 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn mul_scalar(&self, rhs: T) -> Self {
        Self::new(self.m00 * rhs, self.m01 * rhs, self.m11 * rhs)
    }

    /// Multiplies `self` by a scaling vector `scale`.
    ///
    /// This is a faster equivalent to `self * Self::from_diagonal(scale)`.
    ///
    /// This operation is not commutative and does not generally preserve symmetry,
    /// so the result is returned as a [`GMat2`].
    #[inline]
    #[must_use]
    pub fn mul_diagonal_scale(&self, scale: GVec2<T>) -> GMat2<T> {
        GMat2::from_cols(self.col(0) * scale.x, self.col(1) * scale.y)
    }

    /// Divides a symmetric 2x2 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn div_scalar(&self, rhs: T) -> Self {
        Self::new(self.m00.div(rhs), self.m01.div(rhs), self.m11.div(rhs))
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
        self.m00.sub(rhs.m00).abs().num_le(max_abs_diff)
            & self.m01.sub(rhs.m01).abs().num_le(max_abs_diff)
            & self.m11.sub(rhs.m11).abs().num_le(max_abs_diff)
    }

    /// Takes the absolute value of each element in `self`.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self::new(self.m00.abs(), self.m01.abs(), self.m11.abs())
    }
}

impl<T: Float> SymmetricGMat2<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> T::Bool {
        self.m00.is_finite() & self.m01.is_finite() & self.m11.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> T::Bool {
        self.m00.is_nan() | self.m01.is_nan() | self.m11.is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> SymmetricGMat2<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar matrix into a SIMD matrix, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: SymmetricGMat2<T::Element>) -> Self {
        Self::new(
            T::splat(value.m00),
            T::splat(value.m01),
            T::splat(value.m11),
        )
    }

    /// Extracts the i-th lane of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub fn extract(&self, i: usize) -> SymmetricGMat2<T::Element> {
        SymmetricGMat2::new(
            self.m00.extract(i),
            self.m01.extract(i),
            self.m11.extract(i),
        )
    }

    /// Extracts the i-th lane of `self` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub unsafe fn extract_unchecked(&self, i: usize) -> SymmetricGMat2<T::Element> {
        unsafe {
            SymmetricGMat2::new(
                self.m00.extract_unchecked(i),
                self.m01.extract_unchecked(i),
                self.m11.extract_unchecked(i),
            )
        }
    }

    /// Replaces the i-th lane of `self` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    pub fn replace(&mut self, i: usize, value: SymmetricGMat2<T::Element>) {
        self.m00.replace(i, value.m00);
        self.m01.replace(i, value.m01);
        self.m11.replace(i, value.m11);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: SymmetricGMat2<T::Element>) {
        unsafe {
            self.m00.replace_unchecked(i, value.m00);
            self.m01.replace_unchecked(i, value.m01);
            self.m11.replace_unchecked(i, value.m11);
        }
    }
}

/// # Conversion
impl<T: Real> SymmetricGMat2<T> {
    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> SymmetricGMat2<U>
    where
        T: NumCast<U>,
    {
        SymmetricGMat2::new(self.m00.cast(), self.m01.cast(), self.m11.cast())
    }
}

#[cfg(feature = "simd")]
impl SymmetricGMat2<f32> {
    /// Converts `self` to a [`Mat2A`].
    #[inline(always)]
    #[must_use]
    pub const fn to_mat2a(self) -> Mat2A {
        Mat2A::from_cols_array(&self.to_cols_array())
    }
}

#[cfg(feature = "simd")]
impl SymmetricGMat2<f64> {
    /// Converts `self` to a [`Mat2A`].
    #[inline]
    #[must_use]
    pub fn to_mat2a(self) -> Mat2A {
        Mat2A::from_cols_array(&self.cast().to_cols_array())
    }
}

impl<T: Real> Default for SymmetricGMat2<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real + Add<Output = T>> Add for SymmetricGMat2<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: SymmetricGMat2<T>) -> Self {
        SymmetricGMat2::new(
            self.m00.add(rhs.m00),
            self.m01.add(rhs.m01),
            self.m11.add(rhs.m11),
        )
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat2<T>> for SymmetricGMat2<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &SymmetricGMat2<T>) -> Self {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<SymmetricGMat2<T>> for &SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn add(self, rhs: SymmetricGMat2<T>) -> SymmetricGMat2<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat2<T>> for &SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn add(self, rhs: &SymmetricGMat2<T>) -> SymmetricGMat2<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> AddAssign for SymmetricGMat2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: SymmetricGMat2<T>) {
        *self = self.add(rhs);
    }
}

impl<T: Real + AddAssign> AddAssign<&SymmetricGMat2<T>> for SymmetricGMat2<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &SymmetricGMat2<T>) {
        self.add_assign(*rhs);
    }
}

impl<T: Real + Add<Output = T>> Add<GMat2<T>> for SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: GMat2<T>) -> GMat2<T> {
        self.to_mat2().add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat2<T>> for SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: &GMat2<T>) -> GMat2<T> {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<GMat2<T>> for &SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: GMat2<T>) -> GMat2<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat2<T>> for &SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: &GMat2<T>) -> GMat2<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<SymmetricGMat2<T>> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: SymmetricGMat2<T>) -> GMat2<T> {
        self.add(rhs.to_mat2())
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat2<T>> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: &SymmetricGMat2<T>) -> GMat2<T> {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<SymmetricGMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: SymmetricGMat2<T>) -> GMat2<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn add(self, rhs: &SymmetricGMat2<T>) -> GMat2<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub for SymmetricGMat2<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: SymmetricGMat2<T>) -> Self {
        SymmetricGMat2::new(
            self.m00.sub(rhs.m00),
            self.m01.sub(rhs.m01),
            self.m11.sub(rhs.m11),
        )
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat2<T>> for SymmetricGMat2<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat2<T>) -> Self {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<SymmetricGMat2<T>> for &SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn sub(self, rhs: SymmetricGMat2<T>) -> SymmetricGMat2<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat2<T>> for &SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat2<T>) -> SymmetricGMat2<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> SubAssign for SymmetricGMat2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: SymmetricGMat2<T>) {
        *self = self.sub(rhs);
    }
}

impl<T: Real + SubAssign> SubAssign<&SymmetricGMat2<T>> for SymmetricGMat2<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &SymmetricGMat2<T>) {
        self.sub_assign(*rhs);
    }
}

impl<T: Real + Sub<Output = T>> Sub<GMat2<T>> for SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: GMat2<T>) -> GMat2<T> {
        self.to_mat2().sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat2<T>> for SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: &GMat2<T>) -> GMat2<T> {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<GMat2<T>> for &SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: GMat2<T>) -> GMat2<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat2<T>> for &SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: &GMat2<T>) -> GMat2<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<SymmetricGMat2<T>> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: SymmetricGMat2<T>) -> GMat2<T> {
        self.sub(rhs.to_mat2())
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat2<T>> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat2<T>) -> GMat2<T> {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<SymmetricGMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: SymmetricGMat2<T>) -> GMat2<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat2<T>) -> GMat2<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul for SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat2<T>) -> GMat2<T> {
        self.mul_symmetric_mat2(&rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat2<T>> for SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat2<T>) -> GMat2<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<SymmetricGMat2<T>> for &SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat2<T>) -> GMat2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat2<T>> for &SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat2<T>) -> GMat2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat2<T>> for SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: GMat2<T>) -> GMat2<T> {
        self.mul_mat2(&rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat2<T>> for SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: &GMat2<T>) -> GMat2<T> {
        self.mul_mat2(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat2<T>> for &SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: GMat2<T>) -> GMat2<T> {
        self.mul_mat2(&rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat2<T>> for &SymmetricGMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: &GMat2<T>) -> GMat2<T> {
        self.mul_mat2(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<SymmetricGMat2<T>> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat2<T>) -> GMat2<T> {
        GMat2::from_cols(self.mul_vec2(rhs.col(0)), self.mul_vec2(rhs.col(1)))
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat2<T>> for GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat2<T>) -> GMat2<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<SymmetricGMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat2<T>) -> GMat2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat2<T>> for &GMat2<T> {
    type Output = GMat2<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat2<T>) -> GMat2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec2<T>> for SymmetricGMat2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: GVec2<T>) -> GVec2<T> {
        self.mul_vec2(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec2<T>> for SymmetricGMat2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: &GVec2<T>) -> GVec2<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec2<T>> for &SymmetricGMat2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: GVec2<T>) -> GVec2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec2<T>> for &SymmetricGMat2<T> {
    type Output = GVec2<T>;
    #[inline]
    fn mul(self, rhs: &GVec2<T>) -> GVec2<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn mul(self, rhs: T) -> SymmetricGMat2<T> {
        self.mul_scalar(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn mul(self, rhs: &T) -> SymmetricGMat2<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for &SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn mul(self, rhs: T) -> SymmetricGMat2<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for &SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn mul(self, rhs: &T) -> SymmetricGMat2<T> {
        (*self).mul(*rhs)
    }
}

// We cannot implement scalar * matrix generically because of Rust's orphan rules.
macro_rules! impl_scalar_left_mul {
    ($($t:ty),*) => {
        $(
            impl Mul<SymmetricGMat2<$t>> for $t {
                type Output = SymmetricGMat2<$t>;
                #[inline]
                fn mul(self, rhs: SymmetricGMat2<$t>) -> SymmetricGMat2<$t> {
                    rhs.mul_scalar(self)
                }
            }

            impl Mul<&SymmetricGMat2<$t>> for $t {
                type Output = SymmetricGMat2<$t>;
                #[inline]
                fn mul(self, rhs: &SymmetricGMat2<$t>) -> SymmetricGMat2<$t> {
                    self.mul(*rhs)
                }
            }

            impl Mul<SymmetricGMat2<$t>> for &$t {
                type Output = SymmetricGMat2<$t>;
                #[inline]
                fn mul(self, rhs: SymmetricGMat2<$t>) -> SymmetricGMat2<$t> {
                    (*self).mul(rhs)
                }
            }

            impl Mul<&SymmetricGMat2<$t>> for &$t {
                type Output = SymmetricGMat2<$t>;
                #[inline]
                fn mul(self, rhs: &SymmetricGMat2<$t>) -> SymmetricGMat2<$t> {
                    (*self).mul(*rhs)
                }
            }
        )*
    };
}

// TODO: Implement for SIMD types
impl_scalar_left_mul!(f32, f64);

impl<T: Real + Mul<Output = T>> MulAssign<T> for SymmetricGMat2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&T> for SymmetricGMat2<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Div<Output = T>> Div<T> for SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn div(self, rhs: T) -> SymmetricGMat2<T> {
        self.div_scalar(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn div(self, rhs: &T) -> SymmetricGMat2<T> {
        self.div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for &SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn div(self, rhs: T) -> SymmetricGMat2<T> {
        (*self).div(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for &SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn div(self, rhs: &T) -> SymmetricGMat2<T> {
        (*self).div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> DivAssign<T> for SymmetricGMat2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = self.div(rhs);
    }
}

impl<T: Real + Div<Output = T>> DivAssign<&T> for SymmetricGMat2<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.div_assign(*rhs);
    }
}

impl<T: Real + Neg<Output = T>> Neg for SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn neg(self) -> SymmetricGMat2<T> {
        SymmetricGMat2::new(-self.m00, -self.m01, -self.m11)
    }
}

impl<T: Real + Neg<Output = T>> Neg for &SymmetricGMat2<T> {
    type Output = SymmetricGMat2<T>;
    #[inline]
    fn neg(self) -> SymmetricGMat2<T> {
        (*self).neg()
    }
}

impl<T: Real> Sum<SymmetricGMat2<T>> for SymmetricGMat2<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, T: Real> Sum<&'a SymmetricGMat2<T>> for SymmetricGMat2<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl<T: Real> AsRef<[T; 3]> for SymmetricGMat2<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 3] {
        unsafe { &*(self as *const SymmetricGMat2<T> as *const [T; 3]) }
    }
}

impl<T: Real> AsMut<[T; 3]> for SymmetricGMat2<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 3] {
        unsafe { &mut *(self as *mut SymmetricGMat2<T> as *mut [T; 3]) }
    }
}

impl<T: Real> From<SymmetricGMat2<T>> for GMat2<T> {
    #[inline]
    fn from(mat: SymmetricGMat2<T>) -> Self {
        mat.to_mat2()
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for SymmetricGMat2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mat = self.to_mat2();
        if let Some(p) = f.precision() {
            write!(f, "{:.*}", p, mat)
        } else {
            write!(f, "{}", mat)
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for SymmetricGMat2<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(SymmetricGMat2))
            .field("m00", &self.m00)
            .field("m01", &self.m01)
            .field("m11", &self.m11)
            .finish()
    }
}
