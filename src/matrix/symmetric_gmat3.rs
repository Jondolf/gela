use crate::matrix::GMat3;
#[cfg(feature = "simd")]
use crate::matrix::Mat3A;
use crate::vector::GVec3;

use core::{iter::Sum, ops::*};

use gnum::{
    num::{Float, NumCast, Real},
    simd::{MaskLike, Select},
};

/// Creates a symmetric 3x3 matrix from its bottom left triangle, including diagonal elements.
///
/// The elements are in column-major order `mCR`, where `C` is the column index
/// and `R` is the row index.
#[inline(always)]
#[must_use]
pub const fn symmetric_gmat3<T: Real>(
    m00: T,
    m01: T,
    m02: T,
    m11: T,
    m12: T,
    m22: T,
) -> SymmetricGMat3<T> {
    SymmetricGMat3::new(m00, m01, m02, m11, m12, m22)
}

/// The bottom left triangle (including the diagonal) of a symmetric 3x3 column-major matrix.
///
/// This is useful for storing a symmetric 3x3 matrix in a more compact form and performing some
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
#[repr(C)]
pub struct SymmetricGMat3<T: Real> {
    /// The first element of the first column.
    pub m00: T,
    /// The second element of the first column.
    pub m01: T,
    /// The third element of the first column.
    pub m02: T,
    /// The second element of the second column.
    pub m11: T,
    /// The third element of the second column.
    pub m12: T,
    /// The third element of the third column.
    pub m22: T,
}

/// # Constants
impl<T: Real> SymmetricGMat3<T> {
    /// All zeros.
    pub const ZERO: Self = Self::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO);

    /// The 3x3 identity matrix, where all diagonal elements are `1.0` and all off-diagonal elements are `0.0`.
    pub const IDENTITY: Self = Self::new(T::ONE, T::ZERO, T::ZERO, T::ONE, T::ZERO, T::ONE);
}

impl<T: Float> SymmetricGMat3<T> {
    /// All `NAN`.
    pub const NAN: Self = Self::new(T::NAN, T::NAN, T::NAN, T::NAN, T::NAN, T::NAN);
}

/// # Constructors
impl<T: Real> SymmetricGMat3<T> {
    /// Creates a new symmetric 3x3 matrix from its bottom left triangle, including diagonal elements.
    ///
    /// The elements are in column-major order `mCR`, where `C` is the column index
    /// and `R` is the row index.
    #[inline(always)]
    #[must_use]
    pub const fn new(m00: T, m01: T, m02: T, m11: T, m12: T, m22: T) -> Self {
        Self {
            m00,
            m01,
            m02,
            m11,
            m12,
            m22,
        }
    }

    /// Creates a symmetric 3x3 matrix from a `[T; 6]` containing the lower left triangle
    /// of the matrix in column-major order `[m00, m01, m02, m11, m12, m22]`.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(m: [T; 6]) -> Self {
        Self::new(m[0], m[1], m[2], m[3], m[4], m[5])
    }

    /// Creates a `[T; 6]` array containing the lower left triangle of the matrix
    /// in column-major order `[m00, m01, m02, m11, m12, m22]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(&self) -> [T; 6] {
        [self.m00, self.m01, self.m02, self.m11, self.m12, self.m22]
    }

    /// Creates a symmetric 3x3 matrix from the lower left triangle of three column vectors.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_upper()`] for the upper triangular version.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_lower(x_axis: GVec3<T>, y_axis: GVec3<T>, z_axis: GVec3<T>) -> Self {
        Self::new(x_axis.x, x_axis.y, x_axis.z, y_axis.y, y_axis.z, z_axis.z)
    }

    /// Creates a symmetric 3x3 matrix from the upper right triangle of three column vectors.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_lower()`] for the lower triangular version.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_upper(x_axis: GVec3<T>, y_axis: GVec3<T>, z_axis: GVec3<T>) -> Self {
        Self::new(x_axis.x, y_axis.x, z_axis.x, y_axis.y, z_axis.y, z_axis.z)
    }

    /// Creates a symmetric 3x3 matrix from the lower left triangle of a `[T; 9]` array
    /// stored in column major order.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_array_upper()`] for the upper triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_lower(m: &[T; 9]) -> Self {
        Self::new(m[0], m[1], m[2], m[4], m[5], m[8])
    }

    /// Creates a symmetric 3x3 matrix from the upper right triangle of a `[T; 9]` array
    /// stored in column major order.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_array_lower()`] for the lower triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_upper(m: &[T; 9]) -> Self {
        Self::new(m[0], m[3], m[6], m[4], m[7], m[8])
    }

    /// Creates a `[T; 9]` array storing data in column major order.
    #[inline]
    #[must_use]
    pub const fn to_cols_array(&self) -> [T; 9] {
        [
            self.m00, self.m01, self.m02, self.m01, self.m11, self.m12, self.m02, self.m12,
            self.m22,
        ]
    }

    /// Creates a symmetric 3x3 matrix from the lower left triangle of a `[[T; 3]; 3]` 2D array
    /// stored in column major order.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_array_2d_upper()`] for the upper triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d_lower(m: &[[T; 3]; 3]) -> Self {
        Self::from_cols_lower(
            GVec3::from_array(m[0]),
            GVec3::from_array(m[1]),
            GVec3::from_array(m[2]),
        )
    }

    /// Creates a symmetric 3x3 matrix from the upper right triangle of a `[[T; 3]; 3]` 2D array
    /// stored in column major order.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_array_2d_lower()`] for the lower triangular version.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d_upper(m: &[[T; 3]; 3]) -> Self {
        Self::from_cols_upper(
            GVec3::from_array(m[0]),
            GVec3::from_array(m[1]),
            GVec3::from_array(m[2]),
        )
    }

    /// Creates a `[[T; 3]; 3]` 2D array storing data in column major order.
    #[inline]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[T; 3]; 3] {
        [
            [self.m00, self.m01, self.m02],
            [self.m01, self.m11, self.m12],
            [self.m02, self.m12, self.m22],
        ]
    }

    /// Creates a symmetric 3x3 matrix with its diagonal set to `diagonal` and all other entries set to 0.
    #[doc(alias = "scale")]
    #[inline]
    #[must_use]
    pub const fn from_diagonal(diagonal: GVec3<T>) -> Self {
        Self::new(
            diagonal.x,
            T::ZERO,
            T::ZERO,
            diagonal.y,
            T::ZERO,
            diagonal.z,
        )
    }

    /// Creates a symmetric 3x3 matrix from the elements in `if_true` and `if_false`
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
            B::select(boolean, if_true.m11, if_false.m11),
            B::select(boolean, if_true.m12, if_false.m12),
            B::select(boolean, if_true.m22, if_false.m22),
        )
    }

    /// Creates a symmetric 3x3 matrix from the lower left triangle of a 3x3 matrix.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_mat3_upper()`] for the upper triangular version.
    #[inline]
    #[must_use]
    pub const fn from_mat3_lower(mat: GMat3<T>) -> Self {
        Self::from_cols_lower(mat.x_axis, mat.y_axis, mat.z_axis)
    }

    /// Creates a symmetric 3x3 matrix from the upper right triangle of a 3x3 matrix.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_mat3_lower()`] for the lower triangular version.
    #[inline]
    #[must_use]
    pub const fn from_mat3_upper(mat: GMat3<T>) -> Self {
        Self::from_cols_upper(mat.x_axis, mat.y_axis, mat.z_axis)
    }

    /// Creates a symmetric 3x3 matrix from the product `M * Mᵀ` of a 3x3 matrix.
    ///
    /// This is the Gram matrix of the rows of `mat`. For the Gram matrix of the columns,
    /// see [`Self::from_mat3_transpose_mul()`].
    ///
    /// The result is always symmetric and positive semi-definite.
    #[inline]
    #[must_use]
    pub fn from_mat3_mul_transpose(mat: GMat3<T>) -> Self {
        let (r0, r1, r2) = (mat.row(0), mat.row(1), mat.row(2));
        Self::new(
            r0.dot(r0),
            r1.dot(r0),
            r2.dot(r0),
            r1.dot(r1),
            r2.dot(r1),
            r2.dot(r2),
        )
    }

    /// Creates a symmetric 3x3 matrix from the product `Mᵀ * M` of a 3x3 matrix.
    ///
    /// This is the Gram matrix of the columns of `mat`. For the Gram matrix of the rows,
    /// see [`Self::from_mat3_mul_transpose()`].
    ///
    /// The result is always symmetric and positive semi-definite.
    #[inline]
    #[must_use]
    pub fn from_mat3_transpose_mul(mat: GMat3<T>) -> Self {
        let (c0, c1, c2) = (mat.x_axis, mat.y_axis, mat.z_axis);
        Self::new(
            c0.dot(c0),
            c1.dot(c0),
            c2.dot(c0),
            c1.dot(c1),
            c2.dot(c1),
            c2.dot(c2),
        )
    }

    /// Creates a 3x3 matrix from the symmetric 3x3 matrix in `self`.
    #[inline]
    #[must_use]
    pub const fn to_mat3(&self) -> GMat3<T> {
        GMat3::from_cols_array(&self.to_cols_array())
    }

    /// Creates a new symmetric 3x3 matrix from the outer product `v * vᵀ`.
    #[inline(always)]
    #[must_use]
    pub fn from_outer_product(v: GVec3<T>) -> Self {
        Self::new(
            v.x * v.x,
            v.x * v.y,
            v.x * v.z,
            v.y * v.y,
            v.y * v.z,
            v.z * v.z,
        )
    }

    /// Creates a symmetric 3x3 matrix from a slice containing the lower left triangle
    /// of the matrix in column-major order `[m00, m01, m02, m11, m12, m22]`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 6 elements long.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[T]) -> Self {
        Self::new(slice[0], slice[1], slice[2], slice[3], slice[4], slice[5])
    }

    /// Writes the lower left triangle of `self` to a slice in column-major order
    /// `[m00, m01, m02, m11, m12, m22]`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 6 elements long.
    #[inline]
    pub fn write_to_slice(&self, slice: &mut [T]) {
        slice[0] = self.m00;
        slice[1] = self.m01;
        slice[2] = self.m02;
        slice[3] = self.m11;
        slice[4] = self.m12;
        slice[5] = self.m22;
    }

    /// Creates a symmetric 3x3 matrix from the lower left triangle of the first 9 values
    /// in `slice`, stored in column major order.
    ///
    /// The elements above the diagonal are ignored, and mirrored from the ones below it.
    /// See [`Self::from_cols_slice_upper()`] for the upper triangular version.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 9 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice_lower(slice: &[T]) -> Self {
        Self::new(slice[0], slice[1], slice[2], slice[4], slice[5], slice[8])
    }

    /// Creates a symmetric 3x3 matrix from the upper right triangle of the first 9 values
    /// in `slice`, stored in column major order.
    ///
    /// The elements below the diagonal are ignored, and mirrored from the ones above it.
    /// See [`Self::from_cols_slice_lower()`] for the lower triangular version.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 9 elements long.
    #[inline]
    #[must_use]
    pub const fn from_cols_slice_upper(slice: &[T]) -> Self {
        Self::new(slice[0], slice[3], slice[6], slice[4], slice[7], slice[8])
    }

    /// Writes the columns of `self` to the first 9 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 9 elements long.
    #[inline]
    pub fn write_cols_to_slice(&self, slice: &mut [T]) {
        slice[0] = self.m00;
        slice[1] = self.m01;
        slice[2] = self.m02;
        slice[3] = self.m01;
        slice[4] = self.m11;
        slice[5] = self.m12;
        slice[6] = self.m02;
        slice[7] = self.m12;
        slice[8] = self.m22;
    }
}

/// # Operations
impl<T: Real> SymmetricGMat3<T> {
    /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 2.
    #[inline]
    #[must_use]
    pub fn col(&self, index: usize) -> GVec3<T> {
        match index {
            0 => GVec3::new(self.m00, self.m01, self.m02),
            1 => GVec3::new(self.m01, self.m11, self.m12),
            2 => GVec3::new(self.m02, self.m12, self.m22),
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
            0 => GVec3::new(self.m00, self.m01, self.m02),
            1 => GVec3::new(self.m01, self.m11, self.m12),
            2 => GVec3::new(self.m02, self.m12, self.m22),
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the diagonal of `self`.
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> GVec3<T> {
        GVec3::new(self.m00, self.m11, self.m22)
    }

    /// Returns the trace of `self`, the sum of the diagonal elements.
    ///
    /// This is also the sum of the eigenvalues of `self`.
    #[inline]
    #[must_use]
    pub fn trace(&self) -> T {
        self.m00 + self.m11 + self.m22
    }

    /// Returns the adjugate of `self`, the transpose of its cofactor matrix.
    ///
    /// The adjugate of a symmetric matrix is symmetric. It satisfies
    /// `self * self.adjugate() == SymmetricGMat3::IDENTITY * self.determinant()`,
    /// and is the numerator of the inverse.
    #[inline]
    #[must_use]
    pub fn adjugate(&self) -> Self {
        //     [ a d e ]
        // A = | d b f |
        //     [ e f c ]
        let [a, b, c] = [self.m00, self.m11, self.m22];
        let [d, e, f] = [self.m01, self.m02, self.m12];
        Self::new(
            b * c - f * f,
            f * e - d * c,
            d * f - b * e,
            a * c - e * e,
            d * e - a * f,
            a * b - d * d,
        )
    }

    #[inline(always)]
    #[must_use]
    fn adjugate_and_determinant(&self) -> (Self, T) {
        let adjugate = self.adjugate();
        // det(A) = abc + 2def - af^2 - cd^2 - be^2
        let determinant =
            adjugate.m00 * self.m00 + adjugate.m01 * self.m01 + adjugate.m02 * self.m02;
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

    /// Returns the inverse of `self` or `SymmetricGMat3::ZERO` if the matrix is not invertible.
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

    /// Transforms a 3D vector.
    #[inline]
    #[must_use]
    pub fn mul_vec3(&self, rhs: GVec3<T>) -> GVec3<T> {
        let mut res = self.col(0).mul(rhs.x);
        res = res.add(self.col(1).mul(rhs.y));
        res = res.add(self.col(2).mul(rhs.z));
        res
    }

    /// Multiplies two symmetric 3x3 matrices.
    ///
    /// The product of two symmetric matrices is *only* symmetric if the matrices commute,
    /// meaning that `AB == BA`. This is not the case in general, so the result is returned
    /// as a [`GMat3`].
    ///
    /// For a product that is guaranteed to be symmetric, see [`Self::congruence()`].
    #[inline]
    #[must_use]
    pub fn mul_symmetric_mat3(&self, rhs: &Self) -> GMat3<T> {
        GMat3::from_cols(
            self.mul_vec3(rhs.col(0)),
            self.mul_vec3(rhs.col(1)),
            self.mul_vec3(rhs.col(2)),
        )
    }

    /// Multiplies a symmetric 3x3 matrix by a 3x3 matrix.
    ///
    /// The result is not symmetric in general, so it is returned as a [`GMat3`].
    #[inline]
    #[must_use]
    pub fn mul_mat3(&self, rhs: &GMat3<T>) -> GMat3<T> {
        GMat3::from_cols(
            self.mul_vec3(rhs.x_axis),
            self.mul_vec3(rhs.y_axis),
            self.mul_vec3(rhs.z_axis),
        )
    }

    /// Returns the congruence transform `m * self * mᵀ`.
    ///
    /// The result is guaranteed to be symmetric, even if `m` is not.
    ///
    /// If `m` is a rotation matrix, `mᵀ == m⁻¹`, and this re-expresses `self` in a rotated basis.
    /// This can be used to transform an inertia tensor to a different coordinate system, for example.
    #[inline]
    #[must_use]
    pub fn congruence(&self, m: &GMat3<T>) -> Self {
        let (r0, r1, r2) = (m.row(0), m.row(1), m.row(2));
        let (t0, t1, t2) = (self.mul_vec3(r0), self.mul_vec3(r1), self.mul_vec3(r2));
        Self::new(
            r0.dot(t0),
            r1.dot(t0),
            r2.dot(t0),
            r1.dot(t1),
            r2.dot(t1),
            r2.dot(t2),
        )
    }

    /// Returns the congruence transform `skew(v) * self * skew(v)ᵀ`,
    /// where `skew(v)` is the skew-symmetric matrix such that `skew(v) * u == v.cross(u)`.
    ///
    /// The result is guaranteed to be symmetric.
    ///
    /// In physics simulations, this is the lever arm contribution of an inverse inertia tensor
    /// to an effective mass.
    #[doc(alias = "skew_sandwich")]
    #[inline]
    #[must_use]
    pub fn skew_congruence(&self, v: GVec3<T>) -> Self {
        //             [  0 -z  y ]           [ a d e ]
        // skew(v) =   |  z  0 -x |   self =  | d b f |
        //             [ -y  x  0 ]           [ e f c ]
        let (x, y, z) = (v.x, v.y, v.z);
        let [a, b, c] = [self.m00, self.m11, self.m22];
        let [d, e, f] = [self.m01, self.m02, self.m12];

        // Shared between the diagonal elements of `self * skew(v)ᵀ` below.
        let xf = x * f;
        let ye = y * e;
        let zd = z * d;

        let t0 = GVec3::new(ye - zd, y * f - z * b, y * c - z * f);
        let t1 = GVec3::new(z * a - x * e, zd - xf, z * e - x * c);
        let t2 = GVec3::new(x * d - y * a, x * b - y * d, xf - ye);

        Self::new(
            y * t0.z - z * t0.y,
            z * t0.x - x * t0.z,
            x * t0.y - y * t0.x,
            z * t1.x - x * t1.z,
            x * t1.y - y * t1.x,
            x * t2.y - y * t2.x,
        )
    }

    /// Returns the quadratic form `vᵀ * self * v`.
    ///
    /// In physics simulations, for an inverse inertia tensor,
    /// this is the effective mass along a single axis.
    #[doc(alias = "vector_sandwich")]
    #[inline]
    #[must_use]
    pub fn quadratic_form(&self, v: GVec3<T>) -> T {
        v.dot(self.mul_vec3(v))
    }

    /// Adds two symmetric 3x3 matrices.
    #[inline]
    #[must_use]
    pub fn add_symmetric_mat3(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }

    /// Subtracts two symmetric 3x3 matrices.
    #[inline]
    #[must_use]
    pub fn sub_symmetric_mat3(&self, rhs: &Self) -> Self {
        self.sub(rhs)
    }

    /// Adds a symmetric 3x3 matrix and a 3x3 matrix.
    ///
    /// The result is not symmetric in general, so it is returned as a [`GMat3`].
    #[inline]
    #[must_use]
    pub fn add_mat3(&self, rhs: &GMat3<T>) -> GMat3<T> {
        self.to_mat3().add(rhs)
    }

    /// Subtracts a 3x3 matrix from a symmetric 3x3 matrix.
    ///
    /// The result is not symmetric in general, so it is returned as a [`GMat3`].
    #[inline]
    #[must_use]
    pub fn sub_mat3(&self, rhs: &GMat3<T>) -> GMat3<T> {
        self.to_mat3().sub(rhs)
    }

    /// Multiplies a symmetric 3x3 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn mul_scalar(&self, rhs: T) -> Self {
        Self::new(
            self.m00 * rhs,
            self.m01 * rhs,
            self.m02 * rhs,
            self.m11 * rhs,
            self.m12 * rhs,
            self.m22 * rhs,
        )
    }

    /// Multiplies `self` by a scaling vector `scale`.
    ///
    /// This is a faster equivalent to `self * Self::from_diagonal(scale)`.
    ///
    /// This operation is not commutative and does not generally preserve symmetry,
    /// so the result is returned as a [`GMat3`].
    #[inline]
    #[must_use]
    pub fn mul_diagonal_scale(&self, scale: GVec3<T>) -> GMat3<T> {
        GMat3::from_cols(
            self.col(0) * scale.x,
            self.col(1) * scale.y,
            self.col(2) * scale.z,
        )
    }

    /// Divides a symmetric 3x3 matrix by a scalar.
    #[inline]
    #[must_use]
    pub fn div_scalar(&self, rhs: T) -> Self {
        Self::new(
            self.m00.div(rhs),
            self.m01.div(rhs),
            self.m02.div(rhs),
            self.m11.div(rhs),
            self.m12.div(rhs),
            self.m22.div(rhs),
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
            & self.m11.sub(rhs.m11).abs().num_le(max_abs_diff)
            & self.m12.sub(rhs.m12).abs().num_le(max_abs_diff)
            & self.m22.sub(rhs.m22).abs().num_le(max_abs_diff)
    }

    /// Takes the absolute value of each element in `self`.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self::new(
            self.m00.abs(),
            self.m01.abs(),
            self.m02.abs(),
            self.m11.abs(),
            self.m12.abs(),
            self.m22.abs(),
        )
    }
}

impl<T: Float> SymmetricGMat3<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> T::Bool {
        self.m00.is_finite()
            & self.m01.is_finite()
            & self.m02.is_finite()
            & self.m11.is_finite()
            & self.m12.is_finite()
            & self.m22.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> T::Bool {
        self.m00.is_nan()
            | self.m01.is_nan()
            | self.m02.is_nan()
            | self.m11.is_nan()
            | self.m12.is_nan()
            | self.m22.is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> SymmetricGMat3<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar matrix into a SIMD matrix, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: SymmetricGMat3<T::Element>) -> Self {
        Self::new(
            T::splat(value.m00),
            T::splat(value.m01),
            T::splat(value.m02),
            T::splat(value.m11),
            T::splat(value.m12),
            T::splat(value.m22),
        )
    }

    /// Extracts the i-th lane of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub fn extract(&self, i: usize) -> SymmetricGMat3<T::Element> {
        SymmetricGMat3::new(
            self.m00.extract(i),
            self.m01.extract(i),
            self.m02.extract(i),
            self.m11.extract(i),
            self.m12.extract(i),
            self.m22.extract(i),
        )
    }

    /// Extracts the i-th lane of `self` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub unsafe fn extract_unchecked(&self, i: usize) -> SymmetricGMat3<T::Element> {
        unsafe {
            SymmetricGMat3::new(
                self.m00.extract_unchecked(i),
                self.m01.extract_unchecked(i),
                self.m02.extract_unchecked(i),
                self.m11.extract_unchecked(i),
                self.m12.extract_unchecked(i),
                self.m22.extract_unchecked(i),
            )
        }
    }

    /// Replaces the i-th lane of `self` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    pub fn replace(&mut self, i: usize, value: SymmetricGMat3<T::Element>) {
        self.m00.replace(i, value.m00);
        self.m01.replace(i, value.m01);
        self.m02.replace(i, value.m02);
        self.m11.replace(i, value.m11);
        self.m12.replace(i, value.m12);
        self.m22.replace(i, value.m22);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: SymmetricGMat3<T::Element>) {
        unsafe {
            self.m00.replace_unchecked(i, value.m00);
            self.m01.replace_unchecked(i, value.m01);
            self.m02.replace_unchecked(i, value.m02);
            self.m11.replace_unchecked(i, value.m11);
            self.m12.replace_unchecked(i, value.m12);
            self.m22.replace_unchecked(i, value.m22);
        }
    }
}

/// # Conversion
impl<T: Real> SymmetricGMat3<T> {
    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> SymmetricGMat3<U>
    where
        T: NumCast<U>,
    {
        SymmetricGMat3::new(
            self.m00.cast(),
            self.m01.cast(),
            self.m02.cast(),
            self.m11.cast(),
            self.m12.cast(),
            self.m22.cast(),
        )
    }
}

#[cfg(feature = "simd")]
impl SymmetricGMat3<f32> {
    /// Converts `self` to a [`Mat3A`].
    #[inline(always)]
    #[must_use]
    pub const fn to_mat3a(self) -> Mat3A {
        Mat3A::from_cols_array(&self.to_cols_array())
    }
}

#[cfg(feature = "simd")]
impl SymmetricGMat3<f64> {
    /// Converts `self` to a [`Mat3A`].
    #[inline]
    #[must_use]
    pub fn to_mat3a(self) -> Mat3A {
        Mat3A::from_cols_array(&self.cast().to_cols_array())
    }
}

impl<T: Real> Default for SymmetricGMat3<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real + Add<Output = T>> Add for SymmetricGMat3<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: SymmetricGMat3<T>) -> Self {
        SymmetricGMat3::new(
            self.m00.add(rhs.m00),
            self.m01.add(rhs.m01),
            self.m02.add(rhs.m02),
            self.m11.add(rhs.m11),
            self.m12.add(rhs.m12),
            self.m22.add(rhs.m22),
        )
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat3<T>> for SymmetricGMat3<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &SymmetricGMat3<T>) -> Self {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<SymmetricGMat3<T>> for &SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn add(self, rhs: SymmetricGMat3<T>) -> SymmetricGMat3<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat3<T>> for &SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn add(self, rhs: &SymmetricGMat3<T>) -> SymmetricGMat3<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> AddAssign for SymmetricGMat3<T> {
    #[inline]
    fn add_assign(&mut self, rhs: SymmetricGMat3<T>) {
        *self = self.add(rhs);
    }
}

impl<T: Real + AddAssign> AddAssign<&SymmetricGMat3<T>> for SymmetricGMat3<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &SymmetricGMat3<T>) {
        self.add_assign(*rhs);
    }
}

impl<T: Real + Add<Output = T>> Add<GMat3<T>> for SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: GMat3<T>) -> GMat3<T> {
        self.to_mat3().add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat3<T>> for SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: &GMat3<T>) -> GMat3<T> {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<GMat3<T>> for &SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: GMat3<T>) -> GMat3<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GMat3<T>> for &SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: &GMat3<T>) -> GMat3<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<SymmetricGMat3<T>> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: SymmetricGMat3<T>) -> GMat3<T> {
        self.add(rhs.to_mat3())
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat3<T>> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: &SymmetricGMat3<T>) -> GMat3<T> {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<SymmetricGMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: SymmetricGMat3<T>) -> GMat3<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&SymmetricGMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn add(self, rhs: &SymmetricGMat3<T>) -> GMat3<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<GMat3<T>> for SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: GMat3<T>) -> GMat3<T> {
        self.to_mat3().sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat3<T>> for SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: &GMat3<T>) -> GMat3<T> {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<GMat3<T>> for &SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: GMat3<T>) -> GMat3<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GMat3<T>> for &SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: &GMat3<T>) -> GMat3<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<SymmetricGMat3<T>> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: SymmetricGMat3<T>) -> GMat3<T> {
        self.sub(rhs.to_mat3())
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat3<T>> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat3<T>) -> GMat3<T> {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<SymmetricGMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: SymmetricGMat3<T>) -> GMat3<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat3<T>) -> GMat3<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub for SymmetricGMat3<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: SymmetricGMat3<T>) -> Self {
        SymmetricGMat3::new(
            self.m00.sub(rhs.m00),
            self.m01.sub(rhs.m01),
            self.m02.sub(rhs.m02),
            self.m11.sub(rhs.m11),
            self.m12.sub(rhs.m12),
            self.m22.sub(rhs.m22),
        )
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat3<T>> for SymmetricGMat3<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat3<T>) -> Self {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<SymmetricGMat3<T>> for &SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn sub(self, rhs: SymmetricGMat3<T>) -> SymmetricGMat3<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&SymmetricGMat3<T>> for &SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn sub(self, rhs: &SymmetricGMat3<T>) -> SymmetricGMat3<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> SubAssign for SymmetricGMat3<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: SymmetricGMat3<T>) {
        *self = self.sub(rhs);
    }
}

impl<T: Real + SubAssign> SubAssign<&SymmetricGMat3<T>> for SymmetricGMat3<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &SymmetricGMat3<T>) {
        self.sub_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul for SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat3<T>) -> GMat3<T> {
        self.mul_symmetric_mat3(&rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat3<T>> for SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat3<T>) -> GMat3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<SymmetricGMat3<T>> for &SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat3<T>) -> GMat3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat3<T>> for &SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat3<T>) -> GMat3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat3<T>> for SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: GMat3<T>) -> GMat3<T> {
        self.mul_mat3(&rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat3<T>> for SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &GMat3<T>) -> GMat3<T> {
        self.mul_mat3(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GMat3<T>> for &SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: GMat3<T>) -> GMat3<T> {
        self.mul_mat3(&rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GMat3<T>> for &SymmetricGMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &GMat3<T>) -> GMat3<T> {
        self.mul_mat3(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<SymmetricGMat3<T>> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat3<T>) -> GMat3<T> {
        GMat3::from_cols(
            self.mul_vec3(rhs.col(0)),
            self.mul_vec3(rhs.col(1)),
            self.mul_vec3(rhs.col(2)),
        )
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat3<T>> for GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat3<T>) -> GMat3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<SymmetricGMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: SymmetricGMat3<T>) -> GMat3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&SymmetricGMat3<T>> for &GMat3<T> {
    type Output = GMat3<T>;
    #[inline]
    fn mul(self, rhs: &SymmetricGMat3<T>) -> GMat3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec3<T>> for SymmetricGMat3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: GVec3<T>) -> GVec3<T> {
        self.mul_vec3(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec3<T>> for SymmetricGMat3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: &GVec3<T>) -> GVec3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec3<T>> for &SymmetricGMat3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: GVec3<T>) -> GVec3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec3<T>> for &SymmetricGMat3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: &GVec3<T>) -> GVec3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn mul(self, rhs: T) -> SymmetricGMat3<T> {
        self.mul_scalar(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn mul(self, rhs: &T) -> SymmetricGMat3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for &SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn mul(self, rhs: T) -> SymmetricGMat3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for &SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn mul(self, rhs: &T) -> SymmetricGMat3<T> {
        (*self).mul(*rhs)
    }
}

// We cannot implement scalar * matrix generically because of Rust's orphan rules.
macro_rules! impl_scalar_left_mul {
    ($($t:ty),*) => {
        $(
            impl Mul<SymmetricGMat3<$t>> for $t {
                type Output = SymmetricGMat3<$t>;
                #[inline]
                fn mul(self, rhs: SymmetricGMat3<$t>) -> SymmetricGMat3<$t> {
                    rhs.mul_scalar(self)
                }
            }

            impl Mul<&SymmetricGMat3<$t>> for $t {
                type Output = SymmetricGMat3<$t>;
                #[inline]
                fn mul(self, rhs: &SymmetricGMat3<$t>) -> SymmetricGMat3<$t> {
                    self.mul(*rhs)
                }
            }

            impl Mul<SymmetricGMat3<$t>> for &$t {
                type Output = SymmetricGMat3<$t>;
                #[inline]
                fn mul(self, rhs: SymmetricGMat3<$t>) -> SymmetricGMat3<$t> {
                    (*self).mul(rhs)
                }
            }

            impl Mul<&SymmetricGMat3<$t>> for &$t {
                type Output = SymmetricGMat3<$t>;
                #[inline]
                fn mul(self, rhs: &SymmetricGMat3<$t>) -> SymmetricGMat3<$t> {
                    (*self).mul(*rhs)
                }
            }
        )*
    };
}

// TODO: Implement for SIMD types
impl_scalar_left_mul!(f32, f64);

impl<T: Real + Mul<Output = T>> MulAssign<T> for SymmetricGMat3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&T> for SymmetricGMat3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Div<Output = T>> Div<T> for SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn div(self, rhs: T) -> SymmetricGMat3<T> {
        self.div_scalar(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn div(self, rhs: &T) -> SymmetricGMat3<T> {
        self.div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for &SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn div(self, rhs: T) -> SymmetricGMat3<T> {
        (*self).div(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for &SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn div(self, rhs: &T) -> SymmetricGMat3<T> {
        (*self).div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> DivAssign<T> for SymmetricGMat3<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = self.div(rhs);
    }
}

impl<T: Real + Div<Output = T>> DivAssign<&T> for SymmetricGMat3<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.div_assign(*rhs);
    }
}

impl<T: Real + Neg<Output = T>> Neg for SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn neg(self) -> SymmetricGMat3<T> {
        SymmetricGMat3::new(
            -self.m00, -self.m01, -self.m02, -self.m11, -self.m12, -self.m22,
        )
    }
}

impl<T: Real + Neg<Output = T>> Neg for &SymmetricGMat3<T> {
    type Output = SymmetricGMat3<T>;
    #[inline]
    fn neg(self) -> SymmetricGMat3<T> {
        (*self).neg()
    }
}

impl<T: Real> Sum<SymmetricGMat3<T>> for SymmetricGMat3<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, T: Real> Sum<&'a SymmetricGMat3<T>> for SymmetricGMat3<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl<T: Real> AsRef<[T; 6]> for SymmetricGMat3<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 6] {
        unsafe { &*(self as *const SymmetricGMat3<T> as *const [T; 6]) }
    }
}

impl<T: Real> AsMut<[T; 6]> for SymmetricGMat3<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 6] {
        unsafe { &mut *(self as *mut SymmetricGMat3<T> as *mut [T; 6]) }
    }
}

impl<T: Real> From<SymmetricGMat3<T>> for GMat3<T> {
    #[inline]
    fn from(mat: SymmetricGMat3<T>) -> Self {
        mat.to_mat3()
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for SymmetricGMat3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mat = self.to_mat3();
        if let Some(p) = f.precision() {
            write!(f, "{:.*}", p, mat)
        } else {
            write!(f, "{}", mat)
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for SymmetricGMat3<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct(stringify!(SymmetricGMat3))
            .field("m00", &self.m00)
            .field("m01", &self.m01)
            .field("m02", &self.m02)
            .field("m11", &self.m11)
            .field("m12", &self.m12)
            .field("m22", &self.m22)
            .finish()
    }
}
