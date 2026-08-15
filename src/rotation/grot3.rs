use crate::affine::GAffine3;
use crate::matrix::{GMat3, GMat4};
use crate::rotation::{EulerRot, FromEuler, ToEuler};
use crate::vector::{GVec2, GVec3, GVec4};

use core::{
    iter::{Product, Sum},
    ops::*,
};

use gnum::{
    num::{Float, NumCast, Real, ScalarReal},
    simd::{MaskLike, Select},
};

#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

/// Creates a 3D rotation quaternion.
///
/// This should generally not be called manually unless you know what you are doing.
/// Use one of the other constructors instead such as [`from_axis_angle`].
///
/// [`from_axis_angle`]: GRot3::from_axis_angle
#[inline(always)]
#[must_use]
pub const fn gquat<T: Real>(x: T, y: T, z: T, w: T) -> GRot3<T> {
    GRot3::from_xyzw(x, y, z, w)
}

/// A 3D rotation represented as a quaternion.
///
/// The quaternion is intended to be of unit length to represent a valid rotation,
/// but it may become denormalized through error accumulation over successive operations.
/// Users are responsible for normalizing the quaternion when necessary, using methods
/// such as [`normalize`](Self::normalize).
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "zerocopy", derive(FromBytes, Immutable, KnownLayout))]
#[cfg_attr(feature = "cuda", repr(align(16)))]
#[repr(C)]
#[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
#[doc(alias = "GQuat")]
pub struct GRot3<T: Real> {
    /// The X component of the rotation quaternion.
    ///
    /// For a normalized quaternion, this is the X component of the rotation axis
    /// multiplied by `sin(angle / 2)`.
    pub x: T,
    /// The Y component of the rotation quaternion.
    ///
    /// For a normalized quaternion, this is the Y component of the rotation axis
    /// multiplied by `sin(angle / 2)`.
    pub y: T,
    /// The Z component of the rotation quaternion.
    ///
    /// For a normalized quaternion, this is the Z component of the rotation axis
    /// multiplied by `sin(angle / 2)`.
    pub z: T,
    /// The W component of the rotation quaternion.
    ///
    /// For a normalized quaternion, this is `cos(angle / 2)`, representing
    /// the amount of rotation about the axis.
    pub w: T,
}

/// # Basic Number Constants
impl<T: Real> GRot3<T> {
    /// All zeros.
    pub const ZERO: Self = Self::from_xyzw(T::ZERO, T::ZERO, T::ZERO, T::ZERO);

    /// The identity rotation. Corresponds to no rotation.
    pub const IDENTITY: Self = Self::from_xyzw(T::ZERO, T::ZERO, T::ZERO, T::ONE);
}

/// # Float Constants
impl<T: Float> GRot3<T> {
    /// All `NaN`.
    pub const NAN: Self = Self::from_xyzw(T::NAN, T::NAN, T::NAN, T::NAN);
}

/// # Construction
impl<T: Real> GRot3<T> {
    /// Creates a new 3D rotation.
    ///
    /// This should generally not be called manually unless you know what you are doing.
    /// Use one of the other constructors instead such as [`from_axis_angle`].
    ///
    /// [`from_axis_angle`]: GRot3::from_axis_angle
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    #[inline(always)]
    #[must_use]
    pub const fn from_xyzw(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a 3D rotation from the elements in `if_true` and `if_false`, selecting which to use
    /// based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select<B: Select<T>>(boolean: B, if_true: Self, if_false: Self) -> Self {
        Self {
            x: boolean.select(if_true.x, if_false.x),
            y: boolean.select(if_true.y, if_false.y),
            z: boolean.select(if_true.z, if_false.z),
            w: boolean.select(if_true.w, if_false.w),
        }
    }

    /// Creates a new 3D rotation from an array in the form `[x, y, z, w]`.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    #[inline]
    #[must_use]
    pub const fn from_array(a: [T; 4]) -> Self {
        Self::from_xyzw(a[0], a[1], a[2], a[3])
    }

    /// Creates a new 3D rotation from a 4D vector in the form `[x, y, z, w]`.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    #[inline]
    #[must_use]
    pub const fn from_vec4(v: GVec4<T>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: v.w,
        }
    }

    /// Creates a 3D rotation from a slice in the form `[x, y, z, w]`.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    ///
    /// # Panics
    ///
    /// Panics if `slice` length is less than 4.
    #[inline]
    #[must_use]
    pub fn from_slice(slice: &[T]) -> Self {
        Self::from_xyzw(slice[0], slice[1], slice[2], slice[3])
    }

    /// Writes the 3D rotation to an unaligned slice in the form `[x, y, z, w]`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` length is less than 4.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [T]) {
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
        slice[3] = self.w;
    }

    /// Creates a 3D rotation from a normalized rotation `axis` and `angle` (in radians).
    ///
    /// The axis must be a unit vector.
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: GVec3<T>, angle: T) -> Self {
        let (s, c) = (angle * T::HALF).sin_cos_stable();
        let v = axis * s;
        Self::from_xyzw(v.x, v.y, v.z, c)
    }

    /// Creates a 3D rotation that rotates `v.length()` radians around `v.normalize()`.
    ///
    /// `from_scaled_axis(GVec3::ZERO)` results in the identity rotation.
    #[inline]
    #[must_use]
    pub fn from_scaled_axis(v: GVec3<T>) -> Self {
        let length = v.length();
        let mask = length.num_eq(T::ZERO);
        let q = Self::from_axis_angle(v / length, length);
        Self::select(mask, Self::IDENTITY, q)
    }

    /// Creates a 3D rotation from the `angle` (in radians) around the x axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: T) -> Self {
        let (s, c) = (angle * T::HALF).sin_cos_stable();
        Self::from_xyzw(s, T::ZERO, T::ZERO, c)
    }

    /// Creates a 3D rotation from the `angle` (in radians) around the y axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: T) -> Self {
        let (s, c) = (angle * T::HALF).sin_cos_stable();
        Self::from_xyzw(T::ZERO, s, T::ZERO, c)
    }

    /// Creates a 3D rotation from the `angle` (in radians) around the z axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: T) -> Self {
        let (s, c) = (angle * T::HALF).sin_cos_stable();
        Self::from_xyzw(T::ZERO, T::ZERO, s, c)
    }

    /// Creates a 3D rotation from the columns of a 3x3 rotation matrix.
    ///
    /// Note that if the input axes contain scales, shears, or other non-rotation
    /// transformations, the output of this function is ill-defined.
    #[inline]
    #[must_use]
    pub fn from_rotation_axes(x_axis: GVec3<T>, y_axis: GVec3<T>, z_axis: GVec3<T>) -> Self {
        // This is ugly; we want separate scalar and vectorized paths,
        // but we can't do that at compile time without specialization.
        if T::LANES == 1 {
            // Based on https://github.com/microsoft/DirectXMath `XMRot3ernionRotationMatrix`
            let (m00, m01, m02) = x_axis.into();
            let (m10, m11, m12) = y_axis.into();
            let (m20, m21, m22) = z_axis.into();
            if m22.num_le(T::ZERO).all() {
                // x^2 + y^2 >= z^2 + w^2
                let dif10 = m11 - m00;
                let omm22 = T::ONE - m22;
                if dif10.num_le(T::ZERO).all() {
                    // x^2 >= y^2
                    let four_xsq = omm22 - dif10;
                    let inv4x = T::HALF / four_xsq.sqrt();
                    Self::from_xyzw(
                        four_xsq * inv4x,
                        (m01 + m10) * inv4x,
                        (m02 + m20) * inv4x,
                        (m12 - m21) * inv4x,
                    )
                } else {
                    // y^2 >= x^2
                    let four_ysq = omm22 + dif10;
                    let inv4y = T::HALF / four_ysq.sqrt();
                    Self::from_xyzw(
                        (m01 + m10) * inv4y,
                        four_ysq * inv4y,
                        (m12 + m21) * inv4y,
                        (m20 - m02) * inv4y,
                    )
                }
            } else {
                // z^2 + w^2 >= x^2 + y^2
                let sum10 = m11 + m00;
                let opm22 = T::ONE + m22;
                if sum10.num_le(T::ZERO).all() {
                    // z^2 >= w^2
                    let four_zsq = opm22 - sum10;
                    let inv4z = T::HALF / four_zsq.sqrt();
                    Self::from_xyzw(
                        (m02 + m20) * inv4z,
                        (m12 + m21) * inv4z,
                        four_zsq * inv4z,
                        (m01 - m10) * inv4z,
                    )
                } else {
                    // w^2 >= z^2
                    let four_wsq = opm22 + sum10;
                    let inv4w = T::HALF / four_wsq.sqrt();
                    Self::from_xyzw(
                        (m12 - m21) * inv4w,
                        (m20 - m02) * inv4w,
                        (m01 - m10) * inv4w,
                        four_wsq * inv4w,
                    )
                }
            }
        } else {
            // For the vectorized path, we need to compute the possible states of all branches.
            // This is more ALU work than the branching version, but there are a lot of shared terms.
            // Because this handles multiple lanes, it's still a net win over branching.
            // Reference: https://github.com/bepu/bepuphysics2/blob/master/BepuUtilities/Rot3ernionWide.cs

            // TODO: Is this bit-for-bit identical to the scalar path? If not, change the scalar path.
            let m00 = x_axis.x;
            let m01 = x_axis.y;
            let m02 = x_axis.z;
            let m10 = y_axis.x;
            let m11 = y_axis.y;
            let m12 = y_axis.z;
            let m20 = z_axis.x;
            let m21 = z_axis.y;
            let m22 = z_axis.z;

            let one_add_x = T::ONE + m00;
            let one_sub_x = T::ONE - m00;
            let y_add_z = m11 + m22;
            let y_sub_z = m11 - m22;
            let t_x = one_add_x - y_add_z;
            let t_y = one_sub_x + y_sub_z;
            let t_z = one_sub_x - y_sub_z;
            let t_w = one_add_x + y_add_z;

            let use_upper = m22.num_lt(T::ZERO);
            let use_upper_upper = m00.num_gt(m11);
            let use_lower_upper = m00.num_lt(-m11);
            let t = use_upper.select(
                use_upper_upper.select(t_x, t_y),
                use_lower_upper.select(t_z, t_w),
            );
            let xy_add_yx = m01 + m10;
            let yz_sub_zy = m12 - m21;
            let zx_add_xz = m20 + m02;
            let x = use_upper.select(
                use_upper_upper.select(t_x, xy_add_yx),
                use_lower_upper.select(zx_add_xz, yz_sub_zy),
            );
            let yz_add_zy = m12 + m21;
            let zx_sub_xz = m20 - m02;
            let y = use_upper.select(
                use_upper_upper.select(xy_add_yx, t_y),
                use_lower_upper.select(yz_add_zy, zx_sub_xz),
            );
            let xy_sub_yx = m01 - m10;
            let z = use_upper.select(
                use_upper_upper.select(zx_add_xz, yz_add_zy),
                use_lower_upper.select(t_z, xy_sub_yx),
            );
            let w = use_upper.select(
                use_upper_upper.select(yz_sub_zy, zx_sub_xz),
                use_lower_upper.select(xy_sub_yx, t_w),
            );
            let scale = T::HALF / t.sqrt();
            Self::from_xyzw(x * scale, y * scale, z * scale, w * scale)
        }
    }

    /// Creates a 3D rotation from a 3x3 rotation matrix.
    ///
    /// Note that if the input matrix contain scales, shears, or other non-rotation
    /// transformations, the output of this function is ill-defined.
    #[inline]
    #[must_use]
    pub fn from_mat3(mat: &GMat3<T>) -> Self {
        Self::from_rotation_axes(mat.x_axis, mat.y_axis, mat.z_axis)
    }

    /// Creates a 3D rotation from the upper 3x3 rotation matrix inside a homogeneous 4x4 matrix.
    ///
    /// Note that if the upper 3x3 matrix contain scales, shears, or other non-rotation
    /// transformations, the output of this function is ill-defined.
    #[inline]
    #[must_use]
    pub fn from_mat4(mat: &GMat4<T>) -> Self {
        Self::from_rotation_axes(
            mat.x_axis.truncate(),
            mat.y_axis.truncate(),
            mat.z_axis.truncate(),
        )
    }
}

impl<T: ScalarReal> GRot3<T> {
    /// Creates a 3D rotation from the given Euler rotation sequence
    /// and angles (in radians).
    #[inline]
    #[must_use]
    pub fn from_euler(euler: EulerRot, a: T, b: T, c: T) -> Self {
        Self::from_euler_angles(euler, a, b, c)
    }
}

impl<T: Float> GRot3<T> {
    /// Computes the minimal rotation required for transforming `from` into `to`,
    /// such that `GRot3::from_rotation_arc(from, to) * from =≈ to`.
    ///
    /// The rotation is in the plane spanned by the two vectors. This will rotate
    /// at most 180 degrees. The inputs must be unit vectors.
    ///
    /// For near-singular cases (near `from =≈ to` and `from =≈ -to`) the current implementation
    /// is only accurate to about `0.001` (for `f32`).
    #[must_use]
    pub fn from_rotation_arc(from: GVec3<T>, to: GVec3<T>) -> Self {
        let one_minus_eps: T = T::ONE - (T::ONE + T::ONE) * T::EPSILON;

        let dot = from.dot(to);

        // 0° singularity: from ≈= to
        let gt = dot.num_gt(one_minus_eps);

        // 180° singularity: from =≈ -to
        let lt = dot.num_lt(-one_minus_eps);
        let q_lt = Self::from_axis_angle(from.any_orthonormal_vector(), T::PI);

        // General case
        let c = from.cross(to);
        let q_else = Self::from_xyzw(c.x, c.y, c.z, T::ONE + dot).normalize();

        Self::select(gt, Self::IDENTITY, Self::select(lt, q_lt, q_else))
    }

    /// Computes the minimal rotation required for transforming `from` into either `to` or `-to`,
    /// such that `to.dot(GRot3::from_rotation_arc_colinear(from, to) * from).abs() ≈= 1.0`.
    ///
    /// The rotation is in the plane spanned by the two vectors. This will rotate
    /// at most 90 degrees. The inputs must be unit vectors.
    #[inline]
    #[must_use]
    pub fn from_rotation_arc_colinear(from: GVec3<T>, to: GVec3<T>) -> Self {
        let mask = from.dot(to).num_lt(T::ZERO);

        // Specialize scalar and vectorized paths to avoid computing
        // both branches for the scalar case.
        if T::LANES == 1 {
            if mask.all() {
                Self::from_rotation_arc(from, -to)
            } else {
                Self::from_rotation_arc(from, to)
            }
        } else {
            Self::select(
                mask,
                Self::from_rotation_arc(from, -to),
                Self::from_rotation_arc(from, to),
            )
        }
    }

    /// Computes the minimal rotation required for transforming `from` into `to`,
    /// such that `GRot3::from_rotation_arc_2d(from, to) * from =≈ to`.
    ///
    /// The resulting rotation is about the z axis. This will rotate
    /// at most 180 degrees. The inputs must be unit vectors.
    ///
    /// For near-singular cases (near `from =≈ to` and `from =≈ -to`) the current implementation
    /// is only accurate to about `0.001` (for `f32`).
    #[must_use]
    pub fn from_rotation_arc_2d(from: GVec2<T>, to: GVec2<T>) -> Self {
        let one_minus_eps: T = T::ONE - (T::ONE + T::ONE) * T::EPSILON;
        let dot = from.dot(to);

        // Specialize scalar and vectorized paths to avoid computing
        // both branches for the scalar case.
        if T::LANES == 1 {
            if dot.num_gt(one_minus_eps).all() {
                // 0° singularity: from ≈ to
                Self::IDENTITY
            } else if dot.num_lt(-one_minus_eps).all() {
                // 180° singularity: from ≈ -to
                // rotation around z by PI radians
                Self::from_xyzw(T::ZERO, T::ZERO, T::ONE, T::ZERO)
            } else {
                // vector3 cross where z=0
                let z = from.x * to.y - to.x * from.y;
                let w = T::ONE + dot;
                // calculate length with x=0 and y=0 to normalize
                let len_rcp = T::ONE / (z * z + w * w).sqrt();
                Self::from_xyzw(T::ZERO, T::ZERO, z * len_rcp, w * len_rcp)
            }
        } else {
            let gt = dot.num_gt(one_minus_eps);
            let lt = dot.num_lt(-one_minus_eps);
            let q_lt = Self::from_xyzw(T::ZERO, T::ZERO, T::ONE, T::ZERO);
            let z = from.x * to.y - to.x * from.y;
            let w = T::ONE + dot;
            let len_rcp = T::ONE / (z * z + w * w).sqrt();
            let q_else = Self::from_xyzw(T::ZERO, T::ZERO, z * len_rcp, w * len_rcp);
            Self::select(gt, Self::IDENTITY, Self::select(lt, q_lt, q_else))
        }
    }
}

impl<T: Real> GRot3<T> {
    /// Creates a 3D rotation from a facing direction and an up direction.
    ///
    /// This is for a left-handed coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(dir: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_rh(-dir, up)
    }

    /// Creates a 3D rotation from facing direction and an up direction.
    ///
    /// This is for a right-handed coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(dir: GVec3<T>, up: GVec3<T>) -> Self {
        let f = dir;
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self::from_rotation_axes(
            GVec3::new(s.x, u.x, -f.x),
            GVec3::new(s.y, u.y, -f.y),
            GVec3::new(s.z, u.z, -f.z),
        )
    }

    /// Creates a 3D rotation using a camera position, a focal point, and an up direction.
    ///
    /// This is for a left-handed coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: GVec3<T>, center: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_lh(center.sub(eye).normalize(), up)
    }

    /// Creates a 3D rotation using a camera position, a focal point, and an up direction.
    ///
    /// This is for a right-handed coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_at_rh(eye: GVec3<T>, center: GVec3<T>, up: GVec3<T>) -> Self {
        Self::look_to_rh(center.sub(eye).normalize(), up)
    }

    /// Returns the normalized rotation axis and angle (in radians) of `self`.
    #[inline]
    #[must_use]
    pub fn to_axis_angle(self) -> (GVec3<T>, T) {
        let epsilon: T = T::from_f32(1.0e-8);
        let v = GVec3::new(self.x, self.y, self.z);
        let length = v.length();
        let is_non_zero = length.num_ge(epsilon);

        let angle = T::from_f32(2.0) * length.atan2_stable(self.w);
        let axis = v / length;

        (
            GVec3::select(is_non_zero, axis, GVec3::X),
            is_non_zero.select(angle, T::ZERO),
        )
    }

    /// Returns the rotation axis scaled by the rotation in radians.
    #[inline]
    #[must_use]
    pub fn to_scaled_axis(self) -> GVec3<T> {
        let (axis, angle) = self.to_axis_angle();
        axis * angle
    }

    /// Returns the quaternion in `self` as an array in the form `[x, y, z, w]`.
    #[inline]
    #[must_use]
    pub fn to_array(self) -> [T; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Returns the vector part of the quaternion.
    ///
    /// For a normalized quaternion, this is the rotation axis multiplied by `sin(angle / 2)`.
    #[inline]
    #[must_use]
    pub fn xyz(self) -> GVec3<T> {
        GVec3::new(self.x, self.y, self.z)
    }
}

impl<T: ScalarReal> GRot3<T> {
    /// Returns the rotation angles for the given Euler rotation sequence.
    #[inline]
    #[must_use]
    pub fn to_euler(self, order: EulerRot) -> (T, T, T) {
        self.to_euler_angles(order)
    }
}

impl<T: Real> GRot3<T> {
    /// Returns the conjugate of `self`. For a unit quaternion,
    /// the conjugate is also the inverse.
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Returns the inverse of a normalized 3D rotation.
    ///
    /// Typically, the inverse returns the conjugate of a normalized 3D rotation.
    /// Because `self` is assumed to already be unit length, this method *does not*
    /// normalize before returning the conjugate.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        self.conjugate()
    }

    /// Computes the dot product of `self` and `rhs`. The dot product is
    /// equal to the cosine of the angle between two 3D rotations.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> T {
        GVec4::from(self).dot(GVec4::from(rhs))
    }

    /// Computes the length of `self`.
    #[doc(alias = "magnitude")]
    #[inline]
    #[must_use]
    pub fn length(self) -> T {
        GVec4::from(self).length()
    }

    /// Computes the squared length of `self`.
    ///
    /// This is generally faster than `length()` as it avoids a square
    /// root operation.
    #[doc(alias = "magnitude2")]
    #[inline]
    #[must_use]
    pub fn length_squared(self) -> T {
        GVec4::from(self).length_squared()
    }

    /// Computes `1.0 / length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn length_recip(self) -> T {
        GVec4::from(self).length_recip()
    }

    /// Returns `self` normalized to length 1.0.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    // TODO: Benchmark the `normalize_fast()` version, especially for SIMD types.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        Self::from_vec4(GVec4::from(self).normalize())
    }

    /// Returns whether `self` is of length `1.0` or not.
    #[inline]
    #[must_use]
    pub fn is_normalized(self, eps: T) -> T::Bool {
        GVec4::from(self).is_normalized(eps)
    }

    /// Returns whether `self` is near the identity rotation.
    #[inline]
    #[must_use]
    pub fn is_near_identity(self) -> T::Bool {
        // Based on https://github.com/nfrechette/rtm `rtm::quat_near_identity`.
        //
        // Because of floating point precision, we cannot represent very small rotations.
        //
        // The closest f32 to 1.0 that is not 1.0 itself yields:
        // 0.99999994.acos() * 2.0 = 0.000690533954 rad
        //
        // An error threshold of 1.e-6 is used by default.
        // (1.0 - 1.e-6).acos() * 2.0 = 0.00284714461 rad
        // (1.0 - 1.e-7).acos() * 2.0 = 0.00097656250 rad
        //
        // We don't really care about the angle value itself, only if it's close to 0.
        // This will happen whenever quat.w is close to 1.0.
        //
        // If the quat.w is close to -1.0, the angle will be near 2*PI which is close to
        // a negative 0 rotation. By forcing quat.w to be positive, we'll end up with
        // the shortest path.
        //
        // TODO: For f64, use a threshold of (1.0 - 1e-14).acos_stable() * 2.0
        let threshold_angle = T::from_f32(0.002_847_144_6);
        let positive_w_angle = self.w.abs().min(T::ONE).acos_stable() * (T::ONE + T::ONE);
        positive_w_angle.num_lt(threshold_angle)
    }

    /// Returns the angle (in radians) for the minimal rotation
    /// for transforming this 3D rotation into another.
    ///
    /// Both rotations must be normalized.
    #[inline]
    #[must_use]
    pub fn angle_between(self, rhs: Self) -> T {
        // TODO: This loses precision for very small angles.
        self.dot(rhs).abs().min(T::ONE).acos_stable() * (T::ONE + T::ONE)
    }

    /// Rotates towards `rhs` up to `max_angle` (in radians).
    ///
    /// When `max_angle` is `0.0`, the result will be equal to `self`. When `max_angle` is equal to
    /// `self.angle_between(rhs)`, the result will be equal to `rhs`. If `max_angle` is negative,
    /// rotates towards the exact opposite of `rhs`. Will not go past the target.
    ///
    /// Both rotations must be normalized.
    #[inline]
    #[must_use]
    pub fn rotate_towards(self, rhs: Self, max_angle: T) -> Self {
        let angle = self.angle_between(rhs);
        let is_near = angle.num_le(T::from_f32(1e-4));
        let s = (max_angle / angle).clamp(T::NEG_ONE, T::ONE);
        Self::select(is_near, rhs, self.slerp(rhs, s))
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs`
    /// is less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two 3D rotations contain similar elements. It works
    /// best when comparing with a known value. The `max_abs_diff` that should be used used
    /// depends on the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: T) -> T::Bool {
        GVec4::from(self).abs_diff_eq(GVec4::from(rhs), max_abs_diff)
    }

    #[inline(always)]
    #[must_use]
    fn lerp_impl(self, end: Self, s: T) -> Self {
        (self * (T::ONE - s) + end * s).normalize()
    }

    /// Performs a linear interpolation between `self` and `rhs` based on
    /// the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.
    /// When `s` is `1.0`, the result will be equal to `rhs`.
    #[doc(alias = "mix")]
    #[inline]
    #[must_use]
    pub fn lerp(self, end: Self, s: T) -> Self {
        let dot = self.dot(end);
        let bias = dot.num_ge(T::ZERO).select(T::ONE, T::NEG_ONE);
        self.lerp_impl(end * bias, s)
    }

    /// Performs a spherical linear interpolation between `self` and `end`
    /// based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.
    /// When `s` is `1.0`, the result will be equal to `end`.
    #[inline]
    #[must_use]
    pub fn slerp(self, mut end: Self, s: T) -> Self {
        // http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/

        // Note that a rotation can be represented by two quaternions: `q` and
        // `-q`. The slerp path between `q` and `end` will be different from the
        // path between `-q` and `end`. One path will take the long way around and
        // one will take the short way. In order to correct for this, the `dot`
        // product between `self` and `end` should be positive. If the `dot`
        // product is negative, slerp between `self` and `-end`.
        let mut dot = self.dot(end);
        end = Self::select(dot.num_ge(T::ZERO), end, -end);
        dot = dot.abs();

        // TODO: Maybe specialize for floats
        let dot_threshold: T = T::ONE - T::from_f32(f32::EPSILON);

        // Lerp fallback to avoid division by zero
        let use_linear = dot.num_gt(dot_threshold);
        let lerp = self.lerp_impl(end, s);

        // Slerp
        let theta = dot.min(T::ONE).acos_stable();

        let scale1 = (theta * (T::ONE - s)).sin_stable();
        let scale2 = (theta * s).sin_stable();
        let theta_sin = theta.sin_stable();
        let slerp = ((self * scale1) + (end * scale2)) * (T::ONE / theta_sin);

        Self::select(use_linear, lerp, slerp)
    }

    /// Multiplies a 3D rotation and a 3D vector, returning the rotated vector.
    #[inline]
    #[must_use]
    pub fn mul_vec3(self, rhs: GVec3<T>) -> GVec3<T> {
        let w = self.w;
        let b = GVec3::new(self.x, self.y, self.z);
        let b2 = b.dot(b);
        let two = T::ONE + T::ONE;
        rhs.mul(w * w - b2)
            .add(b.mul(rhs.dot(b) * two))
            .add(b.cross(rhs).mul(w * two))
    }

    /// Multiplies two 3D rotations. If they are both normalized, the result will
    /// represent the combined rotation.
    ///
    /// Note that due to floating point rounding, the result may not be perfectly normalized.
    /// Consider normalizing the result after several successive multiplications.
    #[inline]
    #[must_use]
    #[doc(alias = "mul_quat")]
    pub fn mul_rot3(self, rhs: Self) -> Self {
        let (x0, y0, z0, w0) = self.into();
        let (x1, y1, z1, w1) = rhs.into();
        Self::from_xyzw(
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        )
    }

    /// Creates a 3D rotation from a 3x3 rotation matrix inside a 3D affine transform.
    ///
    /// Note that if the input affine matrix contains scales, shears, or other non-rotation
    /// transformations, the output of this function is ill-defined.
    #[inline]
    #[must_use]
    pub fn from_affine3(a: &GAffine3<T>) -> Self {
        Self::from_mat3(&a.matrix3)
    }
}

/// # Float Methods
impl<T: Float> GRot3<T> {
    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> T::Bool {
        GVec4::from(self).is_finite()
    }

    /// Returns `true` if any elements are `NAN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> T::Bool {
        GVec4::from(self).is_nan()
    }
}

/// # SIMD Operations
impl<T: Real> GRot3<T>
where
    T::Element: Real,
{
    /// Broadcasts a scalar 3D rotation into a SIMD 3D rotation, filling all lanes with the same value.
    #[inline]
    pub fn broadcast(value: GRot3<T::Element>) -> Self {
        Self::from_xyzw(
            T::splat(value.x),
            T::splat(value.y),
            T::splat(value.z),
            T::splat(value.w),
        )
    }

    /// Extracts the i-th lane of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub fn extract(&self, i: usize) -> GRot3<T::Element> {
        GRot3::from_xyzw(
            self.x.extract(i),
            self.y.extract(i),
            self.z.extract(i),
            self.w.extract(i),
        )
    }

    /// Extracts the i-th lane of `self` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    #[must_use]
    pub unsafe fn extract_unchecked(&self, i: usize) -> GRot3<T::Element> {
        unsafe {
            GRot3::from_xyzw(
                self.x.extract_unchecked(i),
                self.y.extract_unchecked(i),
                self.z.extract_unchecked(i),
                self.w.extract_unchecked(i),
            )
        }
    }

    /// Replaces the i-th lane of `self` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= T::LANES`.
    #[inline]
    pub fn replace(&mut self, i: usize, value: GRot3<T::Element>) {
        self.x.replace(i, value.x);
        self.y.replace(i, value.y);
        self.z.replace(i, value.z);
        self.w.replace(i, value.w);
    }

    /// Replaces the i-th lane of `self` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Undefined behavior if `i >= T::LANES`.
    #[inline]
    pub unsafe fn replace_unchecked(&mut self, i: usize, value: GRot3<T::Element>) {
        unsafe {
            self.x.replace_unchecked(i, value.x);
            self.y.replace_unchecked(i, value.y);
            self.z.replace_unchecked(i, value.z);
            self.w.replace_unchecked(i, value.w);
        }
    }
}

/// # Conversion
impl<T: Real> GRot3<T> {
    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GRot3<U>
    where
        T: NumCast<U>,
    {
        GRot3::from_xyzw(self.x.cast(), self.y.cast(), self.z.cast(), self.w.cast())
    }
}

impl<T: Real> Default for GRot3<T> {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Real + Add<Output = T>> Add for GRot3<T> {
    type Output = Self;
    /// Adds two quaternions.
    ///
    /// The sum is not guaranteed to be normalized.
    ///
    /// Note that addition is not the same as combining the rotations represented by the
    /// two quaternions! That corresponds to multiplication.
    #[inline]
    fn add(self, rhs: GRot3<T>) -> Self {
        GRot3::from_vec4(GVec4::from(self) + GVec4::from(rhs))
    }
}

impl<T: Real + Add<Output = T>> Add<&GRot3<T>> for GRot3<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &GRot3<T>) -> Self {
        self.add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<GRot3<T>> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn add(self, rhs: GRot3<T>) -> GRot3<T> {
        (*self).add(rhs)
    }
}

impl<T: Real + Add<Output = T>> Add<&GRot3<T>> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn add(self, rhs: &GRot3<T>) -> GRot3<T> {
        (*self).add(*rhs)
    }
}

impl<T: Real + Add<Output = T>> AddAssign for GRot3<T> {
    #[inline]
    fn add_assign(&mut self, rhs: GRot3<T>) {
        *self = self.add(rhs);
    }
}

impl<T: Real + Add<Output = T>> AddAssign<&GRot3<T>> for GRot3<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &GRot3<T>) {
        self.add_assign(*rhs);
    }
}

impl<T: Real + Sub<Output = T>> Sub for GRot3<T> {
    type Output = Self;
    /// Subtracts the `rhs` quaternion from `self`.
    ///
    /// The difference is not guaranteed to be normalized.
    ///
    /// Note that subtraction is not the same as combining the rotations represented by the
    /// two quaternions! That corresponds to multiplication by the inverse.
    #[inline]
    fn sub(self, rhs: GRot3<T>) -> Self {
        GRot3::from_vec4(GVec4::from(self) - GVec4::from(rhs))
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GRot3<T>> for GRot3<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &GRot3<T>) -> Self {
        self.sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<GRot3<T>> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn sub(self, rhs: GRot3<T>) -> GRot3<T> {
        (*self).sub(rhs)
    }
}

impl<T: Real + Sub<Output = T>> Sub<&GRot3<T>> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn sub(self, rhs: &GRot3<T>) -> GRot3<T> {
        (*self).sub(*rhs)
    }
}

impl<T: Real + Sub<Output = T>> SubAssign for GRot3<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: GRot3<T>) {
        *self = self.sub(rhs);
    }
}

impl<T: Real + Sub<Output = T>> SubAssign<&GRot3<T>> for GRot3<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &GRot3<T>) {
        self.sub_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul for GRot3<T> {
    type Output = Self;
    /// Multiplies two 3D rotations. If they are both normalized, the result will
    /// represent the combined rotation.
    ///
    /// Note that due to floating point rounding, the result may not be perfectly normalized.
    /// Consider normalizing the result after several successive multiplications.
    #[inline]
    fn mul(self, rhs: GRot3<T>) -> Self {
        self.mul_rot3(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GRot3<T>> for GRot3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &GRot3<T>) -> Self {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GRot3<T>> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn mul(self, rhs: GRot3<T>) -> GRot3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GRot3<T>> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn mul(self, rhs: &GRot3<T>) -> GRot3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign for GRot3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: GRot3<T>) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&GRot3<T>> for GRot3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &GRot3<T>) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for GRot3<T> {
    type Output = GRot3<T>;
    /// Multiplies a quaternion by a scalar value.
    ///
    /// The product is not guaranteed to be normalized.
    #[inline]
    fn mul(self, rhs: T) -> GRot3<T> {
        Self::from_vec4(GVec4::from(self) * rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GRot3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<T> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn mul(self, rhs: T) -> GRot3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&T> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn mul(self, rhs: &T) -> GRot3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<T> for GRot3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.mul(rhs);
    }
}

impl<T: Real + Mul<Output = T>> MulAssign<&T> for GRot3<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.mul_assign(*rhs);
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec3<T>> for GRot3<T> {
    type Output = GVec3<T>;
    /// Multiplies a 3D rotation and a 3D vector, returning the rotated vector.
    #[inline]
    fn mul(self, rhs: GVec3<T>) -> GVec3<T> {
        self.mul_vec3(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec3<T>> for GRot3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: &GVec3<T>) -> GVec3<T> {
        self.mul(*rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<GVec3<T>> for &GRot3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: GVec3<T>) -> GVec3<T> {
        (*self).mul(rhs)
    }
}

impl<T: Real + Mul<Output = T>> Mul<&GVec3<T>> for &GRot3<T> {
    type Output = GVec3<T>;
    #[inline]
    fn mul(self, rhs: &GVec3<T>) -> GVec3<T> {
        (*self).mul(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for GRot3<T> {
    type Output = Self;
    /// Divides a quaternion by a scalar value.
    ///
    /// The quotient is not guaranteed to be normalized.
    #[inline]
    fn div(self, rhs: T) -> Self {
        Self::from_vec4(GVec4::from(self) / rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for GRot3<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: &T) -> Self {
        self.div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<T> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn div(self, rhs: T) -> GRot3<T> {
        (*self).div(rhs)
    }
}

impl<T: Real + Div<Output = T>> Div<&T> for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn div(self, rhs: &T) -> GRot3<T> {
        (*self).div(*rhs)
    }
}

impl<T: Real + Div<Output = T>> DivAssign<T> for GRot3<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = self.div(rhs);
    }
}

impl<T: Real + Div<Output = T>> DivAssign<&T> for GRot3<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.div_assign(*rhs);
    }
}

impl<T: Real> Neg for GRot3<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self * T::NEG_ONE
    }
}

impl<T: Real> Neg for &GRot3<T> {
    type Output = GRot3<T>;
    #[inline]
    fn neg(self) -> GRot3<T> {
        (*self).neg()
    }
}

impl<T: Real> Sum<GRot3<T>> for GRot3<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, T: Real> Sum<&'a GRot3<T>> for GRot3<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl<T: Real> Product<GRot3<T>> for GRot3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl<'a, T: Real> Product<&'a GRot3<T>> for GRot3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::IDENTITY, |a, &b| a * b)
    }
}

impl<T: Real> From<GRot3<T>> for GVec4<T> {
    #[inline]
    fn from(v: GRot3<T>) -> Self {
        GVec4::new(v.x, v.y, v.z, v.w)
    }
}

impl<T: Real> From<[T; 4]> for GRot3<T> {
    #[inline]
    fn from(comps: [T; 4]) -> Self {
        Self::from_xyzw(comps[0], comps[1], comps[2], comps[3])
    }
}

impl<T: Real> From<GRot3<T>> for [T; 4] {
    #[inline]
    fn from(v: GRot3<T>) -> Self {
        [v.x, v.y, v.z, v.w]
    }
}

impl<T: Real> From<(T, T, T, T)> for GRot3<T> {
    #[inline]
    fn from(comps: (T, T, T, T)) -> Self {
        Self::from_xyzw(comps.0, comps.1, comps.2, comps.3)
    }
}

impl<T: Real> From<GRot3<T>> for (T, T, T, T) {
    #[inline]
    fn from(v: GRot3<T>) -> Self {
        (v.x, v.y, v.z, v.w)
    }
}

impl<T: Real> AsRef<[T; 4]> for GRot3<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 4] {
        unsafe { &*(self as *const GRot3<T> as *const [T; 4]) }
    }
}

impl<T: Real> AsMut<[T; 4]> for GRot3<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 4] {
        unsafe { &mut *(self as *mut GRot3<T> as *mut [T; 4]) }
    }
}

impl<T: Real + core::fmt::Display> core::fmt::Display for GRot3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(p) = f.precision() {
            write!(
                f,
                "[{:.*}, {:.*}, {:.*}, {:.*}]",
                p, self.x, p, self.y, p, self.z, p, self.w
            )
        } else {
            write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
        }
    }
}

impl<T: Real + core::fmt::Debug> core::fmt::Debug for GRot3<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_tuple(stringify!(Rot3))
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .field(&self.w)
            .finish()
    }
}
