use crate::matrix::GMat4;
#[cfg(feature = "simd")]
use crate::matrix::Mat4A;
use crate::vector::GVec4;

use core::{iter::Sum, ops::*};

use gnum::{
    num::{Float, NumCast, Real},
    simd::{MaskLike, Select},
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// Creates a symmetric 4x4 matrix from its bottom left triangle, including diagonal elements.
///
/// The elements are in column-major order `mCR`, where `C` is the column index
/// and `R` is the row index.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
#[must_use]
pub const fn symmetric_gmat4<T: Real>(
    m00: T,
    m01: T,
    m02: T,
    m03: T,
    m11: T,
    m12: T,
    m13: T,
    m22: T,
    m23: T,
    m33: T,
) -> SymmetricGMat4<T> {
    SymmetricGMat4::new(m00, m01, m02, m03, m11, m12, m13, m22, m23, m33)
}

/// The bottom left triangle (including the diagonal) of a symmetric 4x4 column-major matrix.
///
/// This is useful for storing a symmetric 4x4 matrix in a more compact form and performing some
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
pub struct SymmetricGMat4<T: Real> {
    /// The first element of the first column.
    pub m00: T,
    /// The second element of the first column.
    pub m01: T,
    /// The third element of the first column.
    pub m02: T,
    /// The fourth element of the first column.
    pub m03: T,
    /// The second element of the second column.
    pub m11: T,
    /// The third element of the second column.
    pub m12: T,
    /// The fourth element of the second column.
    pub m13: T,
    /// The third element of the third column.
    pub m22: T,
    /// The fourth element of the third column.
    pub m23: T,
    /// The fourth element of the fourth column.
    pub m33: T,
}

/// # Constants
impl<T: Real> SymmetricGMat4<T> {
    /// All zeros.
    pub const ZERO: Self = Self::new(
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
    );

    /// The 4x4 identity matrix, where all diagonal elements are `1.0` and all off-diagonal elements are `0.0`.
    pub const IDENTITY: Self = Self::new(
        T::ONE,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ONE,
        T::ZERO,
        T::ZERO,
        T::ONE,
        T::ZERO,
        T::ONE,
    );
}

impl<T: Float> SymmetricGMat4<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::new(
        T::NAN,
        T::NAN,
        T::NAN,
        T::NAN,
        T::NAN,
        T::NAN,
        T::NAN,
        T::NAN,
        T::NAN,
        T::NAN,
    );
}

/// # Constructors
impl<T: Real> SymmetricGMat4<T> {
    /// Creates a new symmetric 4x4 matrix from its bottom left triangle, including diagonal elements.
    ///
    /// The elements are in column-major order `mCR`, where `C` is the column index
    /// and `R` is the row index.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    #[must_use]
    pub const fn new(
        m00: T,
        m01: T,
        m02: T,
        m03: T,
        m11: T,
        m12: T,
        m13: T,
        m22: T,
        m23: T,
        m33: T,
    ) -> Self {
        Self {
            m00,
            m01,
            m02,
            m03,
            m11,
            m12,
            m13,
            m22,
            m23,
            m33,
        }
    }

    /// Creates a symmetric 4x4 matrix from a `[T; 10]` containing the lower left triangle
    /// of the matrix in column-major order `[m00, m01, m02, m03, m11, m12, m13, m22, m23, m33]`.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(m: [T; 10]) -> Self {
        Self::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9])
    }

    /// Creates a `[T; 10]` array containing the lower left triangle of the matrix
    /// in column-major order `[m00, m01, m02, m03, m11, m12, m13, m22, m23, m33]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(&self) -> [T; 10] {
        [
            self.m00, self.m01, self.m02, self.m03, self.m11, self.m12, self.m13, self.m22,
            self.m23, self.m33,
        ]
    }

    /// Creates a symmetric 4x4 matrix from the lower left triangle of four column vectors.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_upper()`] for the upper triangular version.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_lower(
        x_axis: GVec4<T>,
        y_axis: GVec4<T>,
        z_axis: GVec4<T>,
        w_axis: GVec4<T>,
    ) -> Self {
        Self::new(
            x_axis.x, x_axis.y, x_axis.z, x_axis.w, y_axis.y, y_axis.z, y_axis.w, z_axis.z,
            z_axis.w, w_axis.w,
        )
    }

    /// Creates a symmetric 4x4 matrix from the upper right triangle of four column vectors.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_lower()`] for the lower triangular version.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_upper(
        x_axis: GVec4<T>,
        y_axis: GVec4<T>,
        z_axis: GVec4<T>,
        w_axis: GVec4<T>,
    ) -> Self {
        Self::new(
            x_axis.x, y_axis.x, z_axis.x, w_axis.x, y_axis.y, z_axis.y, w_axis.y, z_axis.z,
            w_axis.z, w_axis.w,
        )
    }

    /// Creates a symmetric 4x4 matrix from the lower left triangle of a `[T; 16]` array
    /// stored in column major order.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_array_upper()`] for the upper triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_lower(m: &[T; 16]) -> Self {
        Self::new(
            m[0], m[1], m[2], m[3], m[5], m[6], m[7], m[10], m[11], m[15],
        )
    }

    /// Creates a symmetric 4x4 matrix from the upper right triangle of a `[T; 16]` array
    /// stored in column major order.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_array_lower()`] for the lower triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_upper(m: &[T; 16]) -> Self {
        Self::new(
            m[0], m[4], m[8], m[12], m[5], m[9], m[13], m[10], m[14], m[15],
        )
    }

    /// Creates a `[T; 16]` array storing data in column major order.
    #[inline]
    #[must_use]
    pub const fn to_cols_array(&self) -> [T; 16] {
        [
            self.m00, self.m01, self.m02, self.m03, self.m01, self.m11, self.m12, self.m13,
            self.m02, self.m12, self.m22, self.m23, self.m03, self.m13, self.m23, self.m33,
        ]
    }

    /// Creates a symmetric 4x4 matrix from the lower left triangle of a `[[T; 4]; 4]` 2D array
    /// stored in column major order.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_array_2d_upper()`] for the upper triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d_lower(m: &[[T; 4]; 4]) -> Self {
        Self::from_cols_lower(
            GVec4::from_array(m[0]),
            GVec4::from_array(m[1]),
            GVec4::from_array(m[2]),
            GVec4::from_array(m[3]),
        )
    }

    /// Creates a symmetric 4x4 matrix from the upper right triangle of a `[[T; 4]; 4]` 2D array
    /// stored in column major order.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_array_2d_lower()`] for the lower triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d_upper(m: &[[T; 4]; 4]) -> Self {
        Self::from_cols_upper(
            GVec4::from_array(m[0]),
            GVec4::from_array(m[1]),
            GVec4::from_array(m[2]),
            GVec4::from_array(m[3]),
        )
    }

    /// Creates a `[[T; 4]; 4]` 2D array storing data in column major order.
    #[inline]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[T; 4]; 4] {
        [
            [self.m00, self.m01, self.m02, self.m03],
            [self.m01, self.m11, self.m12, self.m13],
            [self.m02, self.m12, self.m22, self.m23],
            [self.m03, self.m13, self.m23, self.m33],
        ]
    }

    /// Creates a symmetric 4x4 matrix with its diagonal set to `diagonal` and all other entries set to 0.
    #[doc(alias = "scale")]
    #[inline]
    #[must_use]
    pub const fn from_diagonal(diagonal: GVec4<T>) -> Self {
        Self::new(
            diagonal.x,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            diagonal.y,
            T::ZERO,
            T::ZERO,
            diagonal.z,
            T::ZERO,
            diagonal.w,
        )
    }

    /// Creates a symmetric 4x4 matrix from the elements in `if_true` and `if_false`
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
            B::select(boolean, if_true.m02, if_false.m02),
            B::select(boolean, if_true.m03, if_false.m03),
            B::select(boolean, if_true.m11, if_false.m11),
            B::select(boolean, if_true.m12, if_false.m12),
            B::select(boolean, if_true.m13, if_false.m13),
            B::select(boolean, if_true.m22, if_false.m22),
            B::select(boolean, if_true.m23, if_false.m23),
            B::select(boolean, if_true.m33, if_false.m33),
        )
    }

    /// Creates a symmetric 4x4 matrix from the lower left triangle of a 4x4 matrix.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_mat4_upper()`] for the upper triangular version.
    #[inline]
    #[must_use]
    pub const fn from_mat4_lower(mat: GMat4<T>) -> Self {
        Self::from_cols_lower(mat.x_axis, mat.y_axis, mat.z_axis, mat.w_axis)
    }

    /// Creates a symmetric 4x4 matrix from the upper right triangle of a 4x4 matrix.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_mat4_lower()`] for the lower triangular version.
    #[inline]
    #[must_use]
    pub const fn from_mat4_upper(mat: GMat4<T>) -> Self {
        Self::from_cols_upper(mat.x_axis, mat.y_axis, mat.z_axis, mat.w_axis)
    }

    /// Creates a symmetric 4x4 matrix from the symmetric part `(M + Mᵀ) / 2` of a 4x4 matrix.
    #[inline]
    #[must_use]
    pub fn from_mat4_symmetric_part(mat: GMat4<T>) -> Self {
        Self::new(
            mat.x_axis.x,
            (mat.x_axis.y + mat.y_axis.x) * T::HALF,
            (mat.x_axis.z + mat.z_axis.x) * T::HALF,
            (mat.x_axis.w + mat.w_axis.x) * T::HALF,
            mat.y_axis.y,
            (mat.y_axis.z + mat.z_axis.y) * T::HALF,
            (mat.y_axis.w + mat.w_axis.y) * T::HALF,
            mat.z_axis.z,
            (mat.z_axis.w + mat.w_axis.z) * T::HALF,
            mat.w_axis.w,
        )
    }

    /// Creates a symmetric 4x4 matrix from the product `M * Mᵀ` of a 4x4 matrix.
    ///
    /// This is the Gram matrix of the rows of `mat`. For the Gram matrix of the columns,
    /// see [`Self::from_mat4_transpose_mul()`].
    ///
    /// The result is always symmetric and positive semi-definite.
    #[inline]
    #[must_use]
    pub fn from_mat4_mul_transpose(mat: GMat4<T>) -> Self {
        let (r0, r1, r2, r3) = (mat.row(0), mat.row(1), mat.row(2), mat.row(3));
        Self::new(
            r0.dot(r0),
            r1.dot(r0),
            r2.dot(r0),
            r3.dot(r0),
            r1.dot(r1),
            r2.dot(r1),
            r3.dot(r1),
            r2.dot(r2),
            r3.dot(r2),
            r3.dot(r3),
        )
    }

    /// Creates a symmetric 4x4 matrix from the product `Mᵀ * M` of a 4x4 matrix.
    ///
    /// This is the Gram matrix of the columns of `mat`. For the Gram matrix of the rows,
    /// see [`Self::from_mat4_mul_transpose()`].
    ///
    /// The result is always symmetric and positive semi-definite.
    #[inline]
    #[must_use]
    pub fn from_mat4_transpose_mul(mat: GMat4<T>) -> Self {
        let (c0, c1, c2, c3) = (mat.x_axis, mat.y_axis, mat.z_axis, mat.w_axis);
        Self::new(
            c0.dot(c0),
            c1.dot(c0),
            c2.dot(c0),
            c3.dot(c0),
            c1.dot(c1),
            c2.dot(c1),
            c3.dot(c1),
            c2.dot(c2),
            c3.dot(c2),
            c3.dot(c3),
        )
    }

    /// Creates a 4x4 matrix from the symmetric 4x4 matrix in `self`.
    #[inline]
    #[must_use]
    pub const fn to_mat4(&self) -> GMat4<T> {
        GMat4::from_cols_array(&self.to_cols_array())
    }

    /// Creates a new symmetric 4x4 matrix from the outer product `v * vᵀ`.
    #[inline(always)]
    #[must_use]
    pub fn from_outer_product(v: GVec4<T>) -> Self {
        Self::new(
            v.x * v.x,
            v.x * v.y,
            v.x * v.z,
            v.x * v.w,
            v.y * v.y,
            v.y * v.z,
            v.y * v.w,
            v.z * v.z,
            v.z * v.w,
            v.w * v.w,
        )
    }

    /// Creates a symmetric 4x4 matrix from a slice containing the lower left triangle
    /// of the matrix in column-major order `[m00, m01, m02, m03, m11, m12, m13, m22, m23, m33]`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 10 elements long.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[T]) -> Self {
        Self::new(
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            slice[8], slice[9],
        )
    }

    /// Writes the lower left triangle of `self` to a slice in column-major order
    /// `[m00, m01, m02, m03, m11, m12, m13, m22, m23, m33]`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 10 elements long.
    #[inline]
    pub fn write_to_slice(&self, slice: &mut [T]) {
        slice[0] = self.m00;
        slice[1] = self.m01;
        slice[2] = self.m02;
        slice[3] = self.m03;
        slice[4] = self.m11;
        slice[5] = self.m12;
        slice[6] = self.m13;
        slice[7] = self.m22;
        slice[8] = self.m23;
        slice[9] = self.m33;
    }

    /// Creates a symmetric 4x4 matrix from the lower left triangle of the first 16 values
    /// in `slice`, stored in column major order.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_slice_upper()`] for the upper triangular version.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 16 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice_lower(slice: &[T]) -> Self {
        Self::new(
            slice[0], slice[1], slice[2], slice[3], slice[5], slice[6], slice[7], slice[10],
            slice[11], slice[15],
        )
    }

    /// Creates a symmetric 4x4 matrix from the upper right triangle of the first 16 values
    /// in `slice`, stored in column major order.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_slice_lower()`] for the lower triangular version.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 16 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice_upper(slice: &[T]) -> Self {
        Self::new(
            slice[0], slice[4], slice[8], slice[12], slice[5], slice[9], slice[13], slice[10],
            slice[14], slice[15],
        )
    }

    /// Writes the columns of `self` to the first 16 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 16 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [T]) {
        slice[0] = self.m00;
        slice[1] = self.m01;
        slice[2] = self.m02;
        slice[3] = self.m03;
        slice[4] = self.m01;
        slice[5] = self.m11;
        slice[6] = self.m12;
        slice[7] = self.m13;
        slice[8] = self.m02;
        slice[9] = self.m12;
        slice[10] = self.m22;
        slice[11] = self.m23;
        slice[12] = self.m03;
        slice[13] = self.m13;
        slice[14] = self.m23;
        slice[15] = self.m33;
    }
}

/// # Operations
impl<T: Real> SymmetricGMat4<T> {
    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> GVec4<T> {
        match index {
            0 => GVec4::new(self.m00, self.m01, self.m02, self.m03),
            1 => GVec4::new(self.m01, self.m11, self.m12, self.m13),
            2 => GVec4::new(self.m02, self.m12, self.m22, self.m23),
            3 => GVec4::new(self.m03, self.m13, self.m23, self.m33),
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
        self.col(index)
    }

    /// Returns the diagonal of `self`.
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> GVec4<T> {
        GVec4::new(self.m00, self.m11, self.m22, self.m33)
    }

    /// Returns the trace of `self`, the sum of the diagonal elements.
    ///
    /// This is also the sum of the eigenvalues of `self`.
    #[inline]
    #[must_use]
    pub fn trace(&self) -> T {
        self.m00 + self.m11 + self.m22 + self.m33
    }

    /// Returns the adjugate of `self`, the transpose of its cofactor matrix.
    ///
    /// The adjugate of a symmetric matrix is symmetric. It satisfies
    /// `self * self.adjugate() == SymmetricGMat4::IDENTITY * self.determinant()`,
    /// and is the numerator of the inverse.
    #[inline]
    #[must_use]
    pub fn adjugate(&self) -> Self {
        //     [ a e f g ]
        // A = | e b h i |
        //     | f h c j |
        //     [ g i j d ]
        let [a, b, c, d] = [self.m00, self.m11, self.m22, self.m33];
        let [e, f, g] = [self.m01, self.m02, self.m03];
        let [h, i, j] = [self.m12, self.m13, self.m23];

        // 2x2 minors of the bottom two rows.
        let c01 = f * i - h * g;
        let c02 = f * j - c * g;
        let c03 = f * d - j * g;
        let c12 = h * j - c * i;
        let c13 = h * d - j * i;
        let c23 = c * d - j * j;

        // 2x2 minors of the top two rows. The sixth is equal to `c01` by symmetry.
        let s01 = a * b - e * e;
        let s02 = a * h - f * e;
        let s03 = a * i - g * e;
        let s12 = e * h - f * b;
        let s13 = e * i - g * b;

        Self::new(
            b * c23 - h * c13 + i * c12,
            -e * c23 + h * c03 - i * c02,
            e * c13 - b * c03 + i * c01,
            -e * c12 + b * c02 - h * c01,
            a * c23 - f * c03 + g * c02,
            -a * c13 + e * c03 - g * c01,
            a * c12 - e * c02 + f * c01,
            g * s13 - i * s03 + d * s01,
            -g * s12 + i * s02 - j * s01,
            f * s12 - h * s02 + c * s01,
        )
    }

    #[inline(always)]
    #[must_use]
    fn adjugate_and_determinant(&self) -> (Self, T) {
        let adjugate = self.adjugate();
        let determinant = adjugate.m00 * self.m00
            + adjugate.m01 * self.m01
            + adjugate.m02 * self.m02
            + adjugate.m03 * self.m03;
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

    /// Returns the inverse of `self` or `SymmetricGMat4::ZERO` if the matrix is not invertible.
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

    /// Transforms a 4D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec4(&self, rhs: GVec4<T>) -> GVec4<T> {
        let mut res = self.col(0).mul(rhs.x);
        res = res.add(self.col(1).mul(rhs.y));
        res = res.add(self.col(2).mul(rhs.z));
        res = res.add(self.col(3).mul(rhs.w));
        res
    }

    /// Multiplies two symmetric 4x4 matrices.
    ///
    /// The product of two symmetric matrices is *only* symmetric if the matrices commute,
    /// meaning that `AB == BA`. This is not the case in general, so the result is returned
    /// as a [`GMat4`].
    ///
    /// For a product that is guaranteed to be symmetric, see [`Self::congruence()`].
    #[inline]
    #[must_use]
    pub fn mul_symmetric_mat4(&self, rhs: &Self) -> GMat4<T> {
        GMat4::from_cols(
            self.mul_vec4(rhs.col(0)),
            self.mul_vec4(rhs.col(1)),
            self.mul_vec4(rhs.col(2)),
            self.mul_vec4(rhs.col(3)),
        )
    }

    /// Multiplies a symmetric 4x4 matrix by a 4x4 matrix.
    ///
    /// The result is not symmetric in general, so it is returned as a [`GMat4`].
    #[inline]
    #[must_use]
    pub fn mul_mat4(&self, rhs: &GMat4<T>) -> GMat4<T> {
        GMat4::from_cols(
            self.mul_vec4(rhs.x_axis),
            self.mul_vec4(rhs.y_axis),
            self.mul_vec4(rhs.z_axis),
            self.mul_vec4(rhs.w_axis),
        )
    }

    /// Returns the congruence transform `m * self * mᵀ`.
    ///
    /// The result is guaranteed to be symmetric, even if `m` is not.
    ///
    /// If `m` is a rotation matrix, `mᵀ == m⁻¹`, and this re-expresses `self` in a rotated basis.
    #[inline]
    #[must_use]
    pub fn congruence(&self, m: &GMat4<T>) -> Self {
        let (r0, r1, r2, r3) = (m.row(0), m.row(1), m.row(2), m.row(3));
        let (t0, t1, t2, t3) = (
            self.mul_vec4(r0),
            self.mul_vec4(r1),
            self.mul_vec4(r2),
            self.mul_vec4(r3),
        );
        Self::new(
            r0.dot(t0),
            r1.dot(t0),
            r2.dot(t0),
            r3.dot(t0),
            r1.dot(t1),
            r2.dot(t1),
            r3.dot(t1),
            r2.dot(t2),
            r3.dot(t2),
            r3.dot(t3),
        )
    }

    /// Evaluates the quadratic form of `self` at `v`, returning `vᵀ * self * v`.
    ///
    /// In physics simulations, for an inverse mass matrix,
    /// this is the effective mass along a single axis.
    #[doc(alias = "vector_sandwich")]
    #[inline]
    #[must_use]
    pub fn quadratic_form(&self, v: GVec4<T>) -> T {
        v.dot(self.mul_vec4(v))
    }

    /// Adds two symmetric 4x4 matrices.
    #[inline]
    #[must_use]
    pub fn add_symmetric_mat4(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }

    /// Subtracts two symmetric 4x4 matrices.
    #[inline]
    #[must_use]
    pub fn sub_symmetric_mat4(&self, rhs: &Self) -> Self {
        self.sub(rhs)
    }

    /// Adds a symmetric 4x4 matrix and a 4x4 matrix.
    ///
    /// The result is not symmetric in general, so it is returned as a [`GMat4`].
    #[inline]
    #[must_use]
    pub fn add_mat4(&self, rhs: &GMat4<T>) -> GMat4<T> {
        self.to_mat4().add(rhs)
    }

    /// Subtracts a 4x4 matrix from a symmetric 4x4 matrix.
    ///
    /// The result is not symmetric in general, so it is returned as a [`GMat4`].
    #[inline]
    #[must_use]
    pub fn sub_mat4(&self, rhs: &GMat4<T>) -> GMat4<T> {
        self.to_mat4().sub(rhs)
    }

    /// Multiplies a symmetric 4x4 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn mul_scalar(&self, rhs: T) -> Self {
        Self::new(
            self.m00 * rhs,
            self.m01 * rhs,
            self.m02 * rhs,
            self.m03 * rhs,
            self.m11 * rhs,
            self.m12 * rhs,
            self.m13 * rhs,
            self.m22 * rhs,
            self.m23 * rhs,
            self.m33 * rhs,
        )
    }

    /// Multiplies `self` by a scaling vector `scale`.
    ///
    /// This is a faster equivalent to `self * Self::from_diagonal(scale)`.
    ///
    /// This operation is not commutative and does not generally preserve symmetry,
    /// so the result is returned as a [`GMat4`].
    #[inline]
    #[must_use]
    pub fn mul_diagonal_scale(&self, scale: GVec4<T>) -> GMat4<T> {
        GMat4::from_cols(
            self.col(0) * scale.x,
            self.col(1) * scale.y,
            self.col(2) * scale.z,
            self.col(3) * scale.w,
        )
    }

    /// Divides a symmetric 4x4 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn div_scalar(&self, rhs: T) -> Self {
        Self::new(
            self.m00.div(rhs),
            self.m01.div(rhs),
            self.m02.div(rhs),
            self.m03.div(rhs),
            self.m11.div(rhs),
            self.m12.div(rhs),
            self.m13.div(rhs),
            self.m22.div(rhs),
            self.m23.div(rhs),
            self.m33.div(rhs),
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
        self.m00.sub(rhs.m00).abs().num_le(max_abs_diff)
            & self.m01.sub(rhs.m01).abs().num_le(max_abs_diff)
            & self.m02.sub(rhs.m02).abs().num_le(max_abs_diff)
            & self.m03.sub(rhs.m03).abs().num_le(max_abs_diff)
            & self.m11.sub(rhs.m11).abs().num_le(max_abs_diff)
            & self.m12.sub(rhs.m12).abs().num_le(max_abs_diff)
            & self.m13.sub(rhs.m13).abs().num_le(max_abs_diff)
            & self.m22.sub(rhs.m22).abs().num_le(max_abs_diff)
            & self.m23.sub(rhs.m23).abs().num_le(max_abs_diff)
            & self.m33.sub(rhs.m33).abs().num_le(max_abs_diff)
    }

    /// Takes the absolute value of each element in `self`.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self::new(
            self.m00.abs(),
            self.m01.abs(),
            self.m02.abs(),
            self.m03.abs(),
            self.m11.abs(),
            self.m12.abs(),
            self.m13.abs(),
            self.m22.abs(),
            self.m23.abs(),
            self.m33.abs(),
        )
    }
}

impl<T: Float> SymmetricGMat4<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> T::Bool {
        self.m00.is_finite()
            & self.m01.is_finite()
            & self.m02.is_finite()
            & self.m03.is_finite()
            & self.m11.is_finite()
            & self.m12.is_finite()
            & self.m13.is_finite()
            & self.m22.is_finite()
            & self.m23.is_finite()
            & self.m33.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> T::Bool {
        self.m00.is_nan()
            | self.m01.is_nan()
            | self.m02.is_nan()
            | self.m03.is_nan()
            | self.m11.is_nan()
            | self.m12.is_nan()
            | self.m13.is_nan()
            | self.m22.is_nan()
            | self.m23.is_nan()
            | self.m33.is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> SymmetricGMat4<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar matrix into a SIMD matrix, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: SymmetricGMat4<T::Element>) -> Self {
        Self::new(
            T::splat(value.m00),
            T::splat(value.m01),
            T::splat(value.m02),
            T::splat(value.m03),
            T::splat(value.m11),
            T::splat(value.m12),
            T::splat(value.m13),
            T::splat(value.m22),
            T::splat(value.m23),
            T::splat(value.m33),
        )
    }

    /// Extracts the i-th lane of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub fn extract(&self, i: usize) -> SymmetricGMat4<T::Element> {
        SymmetricGMat4::new(
            self.m00.extract(i),
            self.m01.extract(i),
            self.m02.extract(i),
            self.m03.extract(i),
            self.m11.extract(i),
            self.m12.extract(i),
            self.m13.extract(i),
            self.m22.extract(i),
            self.m23.extract(i),
            self.m33.extract(i),
        )
    }

    /// Extracts the i-th lane of `self` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub unsafe fn extract_unchecked(&self, i: usize) -> SymmetricGMat4<T::Element> {
        unsafe {
            SymmetricGMat4::new(
                self.m00.extract_unchecked(i),
                self.m01.extract_unchecked(i),
                self.m02.extract_unchecked(i),
                self.m03.extract_unchecked(i),
                self.m11.extract_unchecked(i),
                self.m12.extract_unchecked(i),
                self.m13.extract_unchecked(i),
                self.m22.extract_unchecked(i),
                self.m23.extract_unchecked(i),
                self.m33.extract_unchecked(i),
            )
        }
    }

    /// Replaces the i-th lane of `self` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    pub fn replace(&mut self, i: usize, value: SymmetricGMat4<T::Element>) {
        self.m00.replace(i, value.m00);
        self.m01.replace(i, value.m01);
        self.m02.replace(i, value.m02);
        self.m03.replace(i, value.m03);
        self.m11.replace(i, value.m11);
        self.m12.replace(i, value.m12);
        self.m13.replace(i, value.m13);
        self.m22.replace(i, value.m22);
        self.m23.replace(i, value.m23);
        self.m33.replace(i, value.m33);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: SymmetricGMat4<T::Element>) {
        unsafe {
            self.m00.replace_unchecked(i, value.m00);
            self.m01.replace_unchecked(i, value.m01);
            self.m02.replace_unchecked(i, value.m02);
            self.m03.replace_unchecked(i, value.m03);
            self.m11.replace_unchecked(i, value.m11);
            self.m12.replace_unchecked(i, value.m12);
            self.m13.replace_unchecked(i, value.m13);
            self.m22.replace_unchecked(i, value.m22);
            self.m23.replace_unchecked(i, value.m23);
            self.m33.replace_unchecked(i, value.m33);
        }
    }
}

/// # Conversion
impl<T: Real> SymmetricGMat4<T> {
    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> SymmetricGMat4<U>
    where
        T: NumCast<U>,
    {
        SymmetricGMat4::new(
            self.m00.cast(),
            self.m01.cast(),
            self.m02.cast(),
            self.m03.cast(),
            self.m11.cast(),
            self.m12.cast(),
            self.m13.cast(),
            self.m22.cast(),
            self.m23.cast(),
            self.m33.cast(),
        )
    }
}

#[cfg(feature = "simd")]
impl SymmetricGMat4<f32> {
    /// Converts `self` to a [`Mat4A`].
    #[inline(always)]
    #[must_use]
    pub const fn to_mat4a(self) -> Mat4A {
        Mat4A::from_cols_array(&self.to_cols_array())
    }
}

#[cfg(feature = "simd")]
impl SymmetricGMat4<f64> {
    /// Converts `self` to a [`Mat4A`].
    #[inline]
    #[must_use]
    pub fn to_mat4a(self) -> Mat4A {
        Mat4A::from_cols_array(&self.cast().to_cols_array())
    }
}

impl<T: Real> Default for SymmetricGMat4<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real + Add<Output = T>> Add for SymmetricGMat4<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: SymmetricGMat4<T>) -> Self {
        SymmetricGMat4::new(
            self.m00.add(rhs.m00),
            self.m01.add(rhs.m01),
            self.m02.add(rhs.m02),
            self.m03.add(rhs.m03),
            self.m11.add(rhs.m11),
            self.m12.add(rhs.m12),
            self.m13.add(rhs.m13),
            self.m22.add(rhs.m22),
            self.m23.add(rhs.m23),
            self.m33.add(rhs.m33),
        )
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat4<T>> for SymmetricGMat4<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &SymmetricGMat4<T>) -> Self {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<SymmetricGMat4<T>> for &SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn add(self, rhs: SymmetricGMat4<T>) -> SymmetricGMat4<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat4<T>> for &SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn add(self, rhs: &SymmetricGMat4<T>) -> SymmetricGMat4<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> AddAssign for SymmetricGMat4<T> {
    #[inline]
    fn add_assign(&mut self, rhs: SymmetricGMat4<T>) {
        *self = self.add(rhs);
    }
}

impl<T: Real + AddAssign> AddAssign<&SymmetricGMat4<T>> for SymmetricGMat4<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &SymmetricGMat4<T>) {
        self.add_assign(*rhs);
    }
}

impl<T: Real + Add<Output = T>> Add<GMat4<T>> for SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: GMat4<T>) -> GMat4<T> {
        self.to_mat4().add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat4<T>> for SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: &GMat4<T>) -> GMat4<T> {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<GMat4<T>> for &SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: GMat4<T>) -> GMat4<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat4<T>> for &SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: &GMat4<T>) -> GMat4<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<SymmetricGMat4<T>> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: SymmetricGMat4<T>) -> GMat4<T> {
        self.add(rhs.to_mat4())
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat4<T>> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: &SymmetricGMat4<T>) -> GMat4<T> {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<SymmetricGMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: SymmetricGMat4<T>) -> GMat4<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn add(self, rhs: &SymmetricGMat4<T>) -> GMat4<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub for SymmetricGMat4<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: SymmetricGMat4<T>) -> Self {
        SymmetricGMat4::new(
            self.m00.sub(rhs.m00),
            self.m01.sub(rhs.m01),
            self.m02.sub(rhs.m02),
            self.m03.sub(rhs.m03),
            self.m11.sub(rhs.m11),
            self.m12.sub(rhs.m12),
            self.m13.sub(rhs.m13),
            self.m22.sub(rhs.m22),
            self.m23.sub(rhs.m23),
            self.m33.sub(rhs.m33),
        )
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat4<T>> for SymmetricGMat4<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat4<T>) -> Self {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<SymmetricGMat4<T>> for &SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn sub(self, rhs: SymmetricGMat4<T>) -> SymmetricGMat4<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat4<T>> for &SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat4<T>) -> SymmetricGMat4<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> SubAssign for SymmetricGMat4<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: SymmetricGMat4<T>) {
        *self = self.sub(rhs);
    }
}

impl<T: Real + SubAssign> SubAssign<&SymmetricGMat4<T>> for SymmetricGMat4<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &SymmetricGMat4<T>) {
        self.sub_assign(*rhs);
    }
}

impl<T: Real + Sub<Output = T>> Sub<GMat4<T>> for SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: GMat4<T>) -> GMat4<T> {
        self.to_mat4().sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat4<T>> for SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: &GMat4<T>) -> GMat4<T> {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<GMat4<T>> for &SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: GMat4<T>) -> GMat4<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat4<T>> for &SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: &GMat4<T>) -> GMat4<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<SymmetricGMat4<T>> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: SymmetricGMat4<T>) -> GMat4<T> {
        self.sub(rhs.to_mat4())
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat4<T>> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat4<T>) -> GMat4<T> {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<SymmetricGMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: SymmetricGMat4<T>) -> GMat4<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat4<T>) -> GMat4<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul for SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat4<T>) -> GMat4<T> {
        self.mul_symmetric_mat4(&rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat4<T>> for SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat4<T>) -> GMat4<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<SymmetricGMat4<T>> for &SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat4<T>) -> GMat4<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat4<T>> for &SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat4<T>) -> GMat4<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat4<T>> for SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: GMat4<T>) -> GMat4<T> {
        self.mul_mat4(&rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat4<T>> for SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &GMat4<T>) -> GMat4<T> {
        self.mul_mat4(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat4<T>> for &SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: GMat4<T>) -> GMat4<T> {
        self.mul_mat4(&rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat4<T>> for &SymmetricGMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &GMat4<T>) -> GMat4<T> {
        self.mul_mat4(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<SymmetricGMat4<T>> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat4<T>) -> GMat4<T> {
        GMat4::from_cols(
            self.mul_vec4(rhs.col(0)),
            self.mul_vec4(rhs.col(1)),
            self.mul_vec4(rhs.col(2)),
            self.mul_vec4(rhs.col(3)),
        )
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat4<T>> for GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat4<T>) -> GMat4<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<SymmetricGMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat4<T>) -> GMat4<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat4<T>> for &GMat4<T> {
    type Output = GMat4<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat4<T>) -> GMat4<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec4<T>> for SymmetricGMat4<T> {
    type Output = GVec4<T>;
    #[inline]
    fn mul(self, rhs: GVec4<T>) -> GVec4<T> {
        self.mul_vec4(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec4<T>> for SymmetricGMat4<T> {
    type Output = GVec4<T>;
    #[inline]
    fn mul(self, rhs: &GVec4<T>) -> GVec4<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec4<T>> for &SymmetricGMat4<T> {
    type Output = GVec4<T>;
    #[inline]
    fn mul(self, rhs: GVec4<T>) -> GVec4<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec4<T>> for &SymmetricGMat4<T> {
    type Output = GVec4<T>;
    #[inline]
    fn mul(self, rhs: &GVec4<T>) -> GVec4<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn mul(self, rhs: T) -> SymmetricGMat4<T> {
        self.mul_scalar(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn mul(self, rhs: &T) -> SymmetricGMat4<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for &SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn mul(self, rhs: T) -> SymmetricGMat4<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for &SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn mul(self, rhs: &T) -> SymmetricGMat4<T> {
        (*self).mul(*rhs)
    }
}

// We cannot implement scalar * matrix generically because of Rust's orphan rules.
macro_rules! impl_scalar_left_mul {
    ($($t:ty),*) => {
        $(
            impl Mul<SymmetricGMat4<$t>> for $t {
                type Output = SymmetricGMat4<$t>;
                #[inline]
                fn mul(self, rhs: SymmetricGMat4<$t>) -> SymmetricGMat4<$t> {
                    rhs.mul_scalar(self)
                }
            }

            impl Mul<&SymmetricGMat4<$t>> for $t {
                type Output = SymmetricGMat4<$t>;
                #[inline]
                fn mul(self, rhs: &SymmetricGMat4<$t>) -> SymmetricGMat4<$t> {
                    self.mul(*rhs)
                }
            }

            impl Mul<SymmetricGMat4<$t>> for &$t {
                type Output = SymmetricGMat4<$t>;
                #[inline]
                fn mul(self, rhs: SymmetricGMat4<$t>) -> SymmetricGMat4<$t> {
                    (*self).mul(rhs)
                }
            }

            impl Mul<&SymmetricGMat4<$t>> for &$t {
                type Output = SymmetricGMat4<$t>;
                #[inline]
                fn mul(self, rhs: &SymmetricGMat4<$t>) -> SymmetricGMat4<$t> {
                    (*self).mul(*rhs)
                }
            }
        )*
    };
}

// TODO: Implement for SIMD types
impl_scalar_left_mul!(f32, f64);

impl<T: Real + Mul<Output = T>> MulAssign<T> for SymmetricGMat4<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&T> for SymmetricGMat4<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Div<Output = T>> Div<T> for SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn div(self, rhs: T) -> SymmetricGMat4<T> {
        self.div_scalar(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn div(self, rhs: &T) -> SymmetricGMat4<T> {
        self.div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for &SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn div(self, rhs: T) -> SymmetricGMat4<T> {
        (*self).div(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for &SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn div(self, rhs: &T) -> SymmetricGMat4<T> {
        (*self).div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> DivAssign<T> for SymmetricGMat4<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = self.div(rhs);
    }
}

impl<T: Real + Div<Output = T>> DivAssign<&T> for SymmetricGMat4<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.div_assign(*rhs);
    }
}

impl<T: Real + Neg<Output = T>> Neg for SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn neg(self) -> SymmetricGMat4<T> {
        SymmetricGMat4::new(
            -self.m00, -self.m01, -self.m02, -self.m03, -self.m11, -self.m12, -self.m13, -self.m22,
            -self.m23, -self.m33,
        )
    }
}

impl<T: Real + Neg<Output = T>> Neg for &SymmetricGMat4<T> {
    type Output = SymmetricGMat4<T>;
    #[inline]
    fn neg(self) -> SymmetricGMat4<T> {
        (*self).neg()
    }
}

impl<T: Real> Sum<SymmetricGMat4<T>> for SymmetricGMat4<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, T: Real> Sum<&'a SymmetricGMat4<T>> for SymmetricGMat4<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl<T: Real> AsRef<[T; 10]> for SymmetricGMat4<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 10] {
        unsafe { &*(self as *const SymmetricGMat4<T> as *const [T; 10]) }
    }
}

impl<T: Real> AsMut<[T; 10]> for SymmetricGMat4<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 10] {
        unsafe { &mut *(self as *mut SymmetricGMat4<T> as *mut [T; 10]) }
    }
}

impl<T: Real> From<SymmetricGMat4<T>> for GMat4<T> {
    #[inline]
    fn from(mat: SymmetricGMat4<T>) -> Self {
        mat.to_mat4()
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for SymmetricGMat4<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mat = self.to_mat4();
        if let Some(p) = f.precision() {
            write!(f, "{:.*}", p, mat)
        } else {
            write!(f, "{}", mat)
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for SymmetricGMat4<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(SymmetricGMat4))
            .field("m00", &self.m00)
            .field("m01", &self.m01)
            .field("m02", &self.m02)
            .field("m03", &self.m03)
            .field("m11", &self.m11)
            .field("m12", &self.m12)
            .field("m13", &self.m13)
            .field("m22", &self.m22)
            .field("m23", &self.m23)
            .field("m33", &self.m33)
            .finish()
    }
}
