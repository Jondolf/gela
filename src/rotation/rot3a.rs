use crate::affine::GAffine3;
use crate::matrix::{GMat3, GMat4};
use crate::rotation::{EulerRot, GRot3};
use crate::vector::{GVec2, GVec3, Vec3A};

use core::{
    fmt,
    iter::{Product, Sum},
    ops::*,
};

use gimd::f32x4;
use gnum::{
    cmp::NumOrd,
    num::{Float, NumCast, Real, Signed},
    simd::{Reduce, Shuffle4, SimdLike},
};

/// Dot product of two 4-lane registers, matching `GVec4::<f32>::dot`.
#[inline]
fn dot4(a: f32x4, b: f32x4) -> f32 {
    Reduce::reduce_sum_stable(a * b)
}

/// Normalizes a 4-lane register, matching `GVec4::<f32>::normalize`.
#[inline]
fn normalize4(v: f32x4) -> f32x4 {
    let recip = 1.0 / Real::sqrt(dot4(v, v));
    v * f32x4::splat(recip)
}

/// Creates a 3D rotation quaternion with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Rot3`](crate::rotation::Rot3).
///
/// This should generally not be called manually unless you know what you are doing.
/// Use one of the other constructors instead such as [`from_axis_angle`].
///
/// [`from_axis_angle`]: Rot3A::from_axis_angle
#[inline(always)]
#[must_use]
pub const fn rot3a(x: f32, y: f32, z: f32, w: f32) -> Rot3A {
    Rot3A::from_xyzw(x, y, z, w)
}

/// A 3D rotation represented as a quaternion with `f32` components, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for some operations,
/// which can provide better performance than [`Rot3`](crate::rotation::Rot3).
///
/// The quaternion is intended to be of unit length to represent a valid rotation,
/// but it may become denormalized through error accumulation over successive operations.
/// Users are responsible for normalizing the quaternion when necessary, using methods
/// such as [`normalize`](Self::normalize).
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
#[doc(alias = "QuatA")]
#[doc(alias = "Quat")]
pub struct Rot3A(f32x4);

/// # Basic Number Constants
impl Rot3A {
    /// All zeros.
    pub const ZERO: Self = Self::from_xyzw(0.0, 0.0, 0.0, 0.0);

    /// The identity rotation. Corresponds to no rotation.
    pub const IDENTITY: Self = Self::from_xyzw(0.0, 0.0, 0.0, 1.0);

    /// All `NaN`.
    pub const NAN: Self = Self::from_xyzw(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
}

/// # Construction
impl Rot3A {
    /// Creates a new 3D rotation.
    ///
    /// This should generally not be called manually unless you know what you are doing.
    /// Use one of the other constructors instead such as [`from_axis_angle`].
    ///
    /// [`from_axis_angle`]: Rot3A::from_axis_angle
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    #[inline(always)]
    #[must_use]
    pub const fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(f32x4::from_array([x, y, z, w]))
    }

    /// Creates a 3D rotation from the elements in `if_true` and `if_false`, selecting which to use
    /// based on the given `boolean`.
    ///
    /// A true boolean uses the corresponding element from `if_true`, and false
    /// uses the element from `if_false`.
    #[inline]
    #[must_use]
    pub fn select(boolean: bool, if_true: Self, if_false: Self) -> Self {
        if boolean { if_true } else { if_false }
    }

    /// Creates a new 3D rotation from an array in the form `[x, y, z, w]`.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    #[inline]
    #[must_use]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self(f32x4::from_array(a))
    }

    /// Creates a new 3D rotation from a 4D vector in the form `[x, y, z, w]`.
    ///
    /// # Preconditions
    ///
    /// This function does not check if the input is normalized, it is up to the user to
    /// provide normalized input or to normalized the resulting rotation.
    #[inline]
    #[must_use]
    pub const fn from_vec4(v: crate::vector::GVec4<f32>) -> Self {
        Self(f32x4::from_array([v.x, v.y, v.z, v.w]))
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
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::from_xyzw(slice[0], slice[1], slice[2], slice[3])
    }

    /// Writes the 3D rotation to an unaligned slice in the form `[x, y, z, w]`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` length is less than 4.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
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
    pub fn from_axis_angle(axis: GVec3<f32>, angle: f32) -> Self {
        GRot3::from_axis_angle(axis, angle).into()
    }

    /// Creates a 3D rotation from a normalized rotation `axis` and `angle` (in radians).
    ///
    /// The axis must be a unit vector.
    #[inline]
    #[must_use]
    pub fn from_axis_angle_vec3a(axis: Vec3A, angle: f32) -> Self {
        // Bit-identical to `GRot3::<f32>::from_axis_angle`: the axis scaling happens in the
        // register, and the components are only read out to place `c` in the fourth lane.
        let (s, c) = Real::sin_cos_stable(angle * 0.5);
        let v = axis * s;
        Self::from_xyzw(v.x, v.y, v.z, c)
    }

    /// Creates a 3D rotation that rotates `v.length()` radians around `v.normalize()`.
    ///
    /// `from_scaled_axis(GVec3::ZERO)` results in the identity rotation.
    #[inline]
    #[must_use]
    pub fn from_scaled_axis(v: GVec3<f32>) -> Self {
        GRot3::from_scaled_axis(v).into()
    }

    /// Creates a 3D rotation from the `angle` (in radians) around the x axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        GRot3::from_rotation_x(angle).into()
    }

    /// Creates a 3D rotation from the `angle` (in radians) around the y axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        GRot3::from_rotation_y(angle).into()
    }

    /// Creates a 3D rotation from the `angle` (in radians) around the z axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        GRot3::from_rotation_z(angle).into()
    }

    /// Creates a 3D rotation from the columns of a 3x3 rotation matrix.
    ///
    /// Note that if the input axes contain scales, shears, or other non-rotation
    /// transformations, the output of this function is ill-defined.
    #[inline]
    #[must_use]
    pub fn from_rotation_axes(x_axis: GVec3<f32>, y_axis: GVec3<f32>, z_axis: GVec3<f32>) -> Self {
        GRot3::from_rotation_axes(x_axis, y_axis, z_axis).into()
    }

    /// Creates a 3D rotation from a 3x3 rotation matrix.
    ///
    /// Note that if the input matrix contain scales, shears, or other non-rotation
    /// transformations, the output of this function is ill-defined.
    #[inline]
    #[must_use]
    pub fn from_mat3(mat: &GMat3<f32>) -> Self {
        GRot3::from_mat3(mat).into()
    }

    /// Creates a 3D rotation from the upper 3x3 rotation matrix inside a homogeneous 4x4 matrix.
    ///
    /// Note that if the upper 3x3 matrix contain scales, shears, or other non-rotation
    /// transformations, the output of this function is ill-defined.
    #[inline]
    #[must_use]
    pub fn from_mat4(mat: &GMat4<f32>) -> Self {
        GRot3::from_mat4(mat).into()
    }

    /// Creates a 3D rotation from the given Euler rotation sequence
    /// and angles (in radians).
    #[inline]
    #[must_use]
    pub fn from_euler(euler: EulerRot, a: f32, b: f32, c: f32) -> Self {
        GRot3::from_euler(euler, a, b, c).into()
    }

    /// Computes the minimal rotation required for transforming `from` into `to`,
    /// such that `Rot3A::from_rotation_arc(from, to) * from =≈ to`.
    ///
    /// The rotation is in the plane spanned by the two vectors. This will rotate
    /// at most 180 degrees. The inputs must be unit vectors.
    ///
    /// For near-singular cases (near `from =≈ to` and `from =≈ -to`) the current implementation
    /// is only accurate to about `0.001` (for `f32`).
    #[inline]
    #[must_use]
    pub fn from_rotation_arc(from: GVec3<f32>, to: GVec3<f32>) -> Self {
        GRot3::from_rotation_arc(from, to).into()
    }

    /// Computes the minimal rotation required for transforming `from` into either `to` or `-to`,
    /// such that `to.dot(Rot3A::from_rotation_arc_colinear(from, to) * from).abs() ≈= 1.0`.
    ///
    /// The rotation is in the plane spanned by the two vectors. This will rotate
    /// at most 90 degrees. The inputs must be unit vectors.
    #[inline]
    #[must_use]
    pub fn from_rotation_arc_colinear(from: GVec3<f32>, to: GVec3<f32>) -> Self {
        GRot3::from_rotation_arc_colinear(from, to).into()
    }

    /// Computes the minimal rotation required for transforming `from` into `to`,
    /// such that `Rot3A::from_rotation_arc_2d(from, to) * from =≈ to`.
    ///
    /// The resulting rotation is about the z axis. This will rotate
    /// at most 180 degrees. The inputs must be unit vectors.
    ///
    /// For near-singular cases (near `from =≈ to` and `from =≈ -to`) the current implementation
    /// is only accurate to about `0.001` (for `f32`).
    #[inline]
    #[must_use]
    pub fn from_rotation_arc_2d(from: GVec2<f32>, to: GVec2<f32>) -> Self {
        GRot3::from_rotation_arc_2d(from, to).into()
    }

    /// Creates a 3D rotation from a facing direction and an up direction.
    ///
    /// This is for a left-handed coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_to_lh(dir: GVec3<f32>, up: GVec3<f32>) -> Self {
        GRot3::look_to_lh(dir, up).into()
    }

    /// Creates a 3D rotation from facing direction and an up direction.
    ///
    /// This is for a right-handed coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_to_rh(dir: GVec3<f32>, up: GVec3<f32>) -> Self {
        GRot3::look_to_rh(dir, up).into()
    }

    /// Creates a 3D rotation using a camera position, a focal point, and an up direction.
    ///
    /// This is for a left-handed coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    #[must_use]
    pub fn look_at_lh(eye: GVec3<f32>, center: GVec3<f32>, up: GVec3<f32>) -> Self {
        GRot3::look_at_lh(eye, center, up).into()
    }

    /// Creates a 3D rotation using a camera position, a focal point, and an up direction.
    ///
    /// This is for a right-handed coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    #[must_use]
    pub fn look_at_rh(eye: GVec3<f32>, center: GVec3<f32>, up: GVec3<f32>) -> Self {
        GRot3::look_at_rh(eye, center, up).into()
    }

    /// Creates a 3D rotation from a 3x3 rotation matrix inside a 3D affine transform.
    ///
    /// Note that if the input affine matrix contains scales, shears, or other non-rotation
    /// transformations, the output of this function is ill-defined.
    #[inline]
    #[must_use]
    pub fn from_affine3(a: &GAffine3<f32>) -> Self {
        GRot3::from_affine3(a).into()
    }
}

impl Rot3A {
    /// Returns the normalized rotation axis and angle (in radians) of `self`.
    #[inline]
    #[must_use]
    pub fn to_axis_angle(self) -> (GVec3<f32>, f32) {
        self.to_quat().to_axis_angle()
    }

    /// Returns the rotation axis scaled by the rotation in radians.
    #[inline]
    #[must_use]
    pub fn to_scaled_axis(self) -> GVec3<f32> {
        self.to_quat().to_scaled_axis()
    }

    /// Returns the quaternion in `self` as an array in the form `[x, y, z, w]`.
    #[inline]
    #[must_use]
    pub fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Returns the vector part of the quaternion.
    ///
    /// For a normalized quaternion, this is the rotation axis multiplied by `sin(angle / 2)`.
    #[inline]
    #[must_use]
    pub fn xyz(self) -> GVec3<f32> {
        GVec3::new(self.x, self.y, self.z)
    }

    /// Returns the rotation angles for the given Euler rotation sequence.
    #[inline]
    #[must_use]
    pub fn to_euler(self, order: EulerRot) -> (f32, f32, f32) {
        self.to_quat().to_euler(order)
    }
}

/// # Operations
impl Rot3A {
    /// Returns the conjugate of `self`. For a unit quaternion,
    /// the conjugate is also the inverse.
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        // `(0, 0, 0, 2w) - self` yields `(-x, -y, -z, w)`
        let w2 = self.w + self.w;
        Self(f32x4::from_array([0.0, 0.0, 0.0, w2]) - self.0)
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
    pub fn dot(self, rhs: Self) -> f32 {
        dot4(self.0, rhs.0)
    }

    /// Computes the length of `self`.
    #[doc(alias = "magnitude")]
    #[inline]
    #[must_use]
    pub fn length(self) -> f32 {
        Real::sqrt(self.length_squared())
    }

    /// Computes the squared length of `self`.
    ///
    /// This is generally faster than `length()` as it avoids a square
    /// root operation.
    #[doc(alias = "magnitude2")]
    #[inline]
    #[must_use]
    pub fn length_squared(self) -> f32 {
        dot4(self.0, self.0)
    }

    /// Computes `1.0 / length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn length_recip(self) -> f32 {
        1.0 / self.length()
    }

    /// Returns `self` normalized to length 1.0.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        Self(normalize4(self.0))
    }

    /// Returns whether `self` is of length `1.0` or not.
    #[inline]
    #[must_use]
    pub fn is_normalized(self, eps: f32) -> bool {
        Signed::abs(self.length_squared() - 1.0) <= eps
    }

    /// Returns whether `self` is near the identity rotation.
    #[inline]
    #[must_use]
    pub fn is_near_identity(self) -> bool {
        // See `GRot3::<f32>::is_near_identity` for where the threshold comes from.
        let threshold_angle = 0.002_847_144_6;
        let positive_w_angle = Real::acos_stable(NumOrd::min(Signed::abs(self.w), 1.0)) * 2.0;
        positive_w_angle < threshold_angle
    }

    /// Returns the angle (in radians) for the minimal rotation
    /// for transforming this 3D rotation into another.
    ///
    /// Both rotations must be normalized.
    #[inline]
    #[must_use]
    pub fn angle_between(self, rhs: Self) -> f32 {
        Real::acos_stable(NumOrd::min(Signed::abs(self.dot(rhs)), 1.0)) * 2.0
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
    pub fn rotate_towards(self, rhs: Self, max_angle: f32) -> Self {
        let angle = self.angle_between(rhs);
        let s = NumOrd::clamp(max_angle / angle, -1.0, 1.0);
        if angle <= 1e-4 {
            rhs
        } else {
            self.slerp(rhs, s)
        }
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
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        let d = Signed::abs(self.0 - rhs.0);
        SimdLike::extract(&d, 0) <= max_abs_diff
            && SimdLike::extract(&d, 1) <= max_abs_diff
            && SimdLike::extract(&d, 2) <= max_abs_diff
            && SimdLike::extract(&d, 3) <= max_abs_diff
    }

    /// Performs a linear interpolation between `self` and `rhs` based on
    /// the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.
    /// When `s` is `1.0`, the result will be equal to `rhs`.
    #[doc(alias = "mix")]
    #[inline]
    #[must_use]
    pub fn lerp(self, end: Self, s: f32) -> Self {
        let bias = if dot4(self.0, end.0) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let end = end.0 * f32x4::splat(bias);
        Self(normalize4(self.lerp_blend(end, s)))
    }

    /// Performs a spherical linear interpolation between `self` and `end`
    /// based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.
    /// When `s` is `1.0`, the result will be equal to `end`.
    #[inline]
    #[must_use]
    pub fn slerp(self, end: Self, s: f32) -> Self {
        let dot = dot4(self.0, end.0);
        let end = if dot >= 0.0 {
            end.0
        } else {
            end.0 * f32x4::splat(-1.0)
        };
        let dot = Signed::abs(dot);

        let dot_threshold = 1.0 - f32::EPSILON;

        // Near-parallel, lerp
        if dot > dot_threshold {
            return Self(normalize4(self.lerp_blend(end, s)));
        }

        // Slerp
        let theta = Real::acos_stable(NumOrd::min(dot, 1.0));

        let angles = f32x4::splat(theta) * f32x4::from_array([1.0 - s, s, 1.0, 1.0]);
        let sins = Real::sin_stable(angles);
        let scale1 = Shuffle4::shuffle::<0, 0, 0, 0>(sins);
        let scale2 = Shuffle4::shuffle::<1, 1, 1, 1>(sins);
        let theta_sin = Shuffle4::shuffle::<2, 2, 2, 2>(sins);
        let recip = f32x4::splat(1.0) / theta_sin;

        Self((self.0 * scale1 + end * scale2) * recip)
    }

    #[inline(always)]
    fn lerp_blend(self, end: f32x4, s: f32) -> f32x4 {
        self.0 * f32x4::splat(1.0 - s) + end * f32x4::splat(s)
    }

    /// Multiplies a 3D rotation and a 3D vector, returning the rotated vector.
    #[inline]
    #[must_use]
    pub fn mul_vec3(self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.to_quat().mul_vec3(rhs)
    }

    /// Multiplies a 3D rotation and a SIMD-aligned 3D vector, returning the rotated vector.
    #[inline]
    #[must_use]
    pub fn mul_vec3a(self, rhs: Vec3A) -> Vec3A {
        let w = self.w;
        let b = Vec3A::from_register(self.0);
        let b2 = b.dot(b);
        rhs * (w * w - b2) + b * (rhs.dot(b) * 2.0) + b.cross(rhs) * (w * 2.0)
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
        let l = self.0; // [x0, y0, z0, w0]
        let r = rhs.0; // [x1, y1, z1, w1]

        let x0 = Shuffle4::shuffle::<0, 0, 0, 0>(l);
        let y0 = Shuffle4::shuffle::<1, 1, 1, 1>(l);
        let z0 = Shuffle4::shuffle::<2, 2, 2, 2>(l);
        let w0 = Shuffle4::shuffle::<3, 3, 3, 3>(l);

        // Sign-adjusted shuffles of `rhs`, one per non-`w` component of `self`.
        let rx = Shuffle4::shuffle::<3, 2, 1, 0>(r) * f32x4::from_array([1.0, -1.0, 1.0, -1.0]);
        let ry = Shuffle4::shuffle::<2, 3, 0, 1>(r) * f32x4::from_array([1.0, 1.0, -1.0, -1.0]);
        let rz = Shuffle4::shuffle::<1, 0, 3, 2>(r) * f32x4::from_array([-1.0, 1.0, 1.0, -1.0]);

        Self(w0 * r + x0 * rx + y0 * ry + z0 * rz)
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        Float::is_finite(self.x)
            && Float::is_finite(self.y)
            && Float::is_finite(self.z)
            && Float::is_finite(self.w)
    }

    /// Returns `true` if any elements are `NAN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        Float::is_nan(self.x)
            || Float::is_nan(self.y)
            || Float::is_nan(self.z)
            || Float::is_nan(self.w)
    }
}

/// # Conversions
impl Rot3A {
    /// Creates a [`Rot3A`] from a [`GRot3<f32>`].
    #[inline(always)]
    #[must_use]
    pub const fn from_quat(q: GRot3<f32>) -> Self {
        Self(f32x4::from_array([q.x, q.y, q.z, q.w]))
    }

    /// Converts `self` to a [`GRot3<f32>`].
    #[inline(always)]
    #[must_use]
    pub fn to_quat(self) -> GRot3<f32> {
        GRot3::from_xyzw(self.x, self.y, self.z, self.w)
    }

    /// Casts the elements of `self` to another type.
    #[inline]
    #[must_use]
    pub fn cast<U: Real>(self) -> GRot3<U>
    where
        f32: NumCast<U>,
    {
        self.to_quat().cast()
    }
}

impl Rot3A {
    /// Returns the backing register, in `x, y, z, w` lane order.
    #[inline(always)]
    pub(crate) const fn register(self) -> f32x4 {
        self.0
    }
}

impl Deref for Rot3A {
    type Target = crate::deref::Vec4<f32>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl DerefMut for Rot3A {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl PartialEq for Rot3A {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z && self.w == rhs.w
    }
}

impl Default for Rot3A {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl fmt::Debug for Rot3A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Rot3A")
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .field(&self.w)
            .finish()
    }
}

impl fmt::Display for Rot3A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl Add for Rot3A {
    type Output = Self;
    /// Adds two quaternions.
    ///
    /// The sum is not guaranteed to be normalized.
    ///
    /// Note that addition is not the same as combining the rotations represented by the
    /// two quaternions! That corresponds to multiplication.
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Rot3A {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs);
    }
}

impl Sub for Rot3A {
    type Output = Self;
    /// Subtracts the `rhs` quaternion from `self`.
    ///
    /// The difference is not guaranteed to be normalized.
    ///
    /// Note that subtraction is not the same as combining the rotations represented by the
    /// two quaternions! That corresponds to multiplication by the inverse.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Rot3A {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(rhs);
    }
}

impl Mul for Rot3A {
    type Output = Self;
    /// Multiplies two 3D rotations. If they are both normalized, the result will
    /// represent the combined rotation.
    ///
    /// Note that due to floating point rounding, the result may not be perfectly normalized.
    /// Consider normalizing the result after several successive multiplications.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_rot3(rhs)
    }
}

impl MulAssign for Rot3A {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs);
    }
}

impl Mul<f32> for Rot3A {
    type Output = Self;
    /// Multiplies a quaternion by a scalar value.
    ///
    /// The product is not guaranteed to be normalized.
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self(self.0 * f32x4::splat(rhs))
    }
}

impl MulAssign<f32> for Rot3A {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = self.mul(rhs);
    }
}

impl Mul<GVec3<f32>> for Rot3A {
    type Output = GVec3<f32>;
    /// Multiplies a 3D rotation and a 3D vector, returning the rotated vector.
    #[inline]
    fn mul(self, rhs: GVec3<f32>) -> GVec3<f32> {
        self.mul_vec3(rhs)
    }
}

impl Mul<Vec3A> for Rot3A {
    type Output = Vec3A;
    /// Multiplies a 3D rotation and a SIMD-aligned 3D vector, returning the rotated vector.
    #[inline]
    fn mul(self, rhs: Vec3A) -> Vec3A {
        self.mul_vec3a(rhs)
    }
}

impl Div<f32> for Rot3A {
    type Output = Self;
    /// Divides a quaternion by a scalar value.
    ///
    /// The quotient is not guaranteed to be normalized.
    #[inline]
    fn div(self, rhs: f32) -> Self {
        Self(self.0 / f32x4::splat(rhs))
    }
}

impl DivAssign<f32> for Rot3A {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self = self.div(rhs);
    }
}

impl Neg for Rot3A {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self * -1.0
    }
}

impl Sum for Rot3A {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl Product for Rot3A {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::IDENTITY, |a, b| a * b)
    }
}

impl From<GRot3<f32>> for Rot3A {
    #[inline(always)]
    fn from(q: GRot3<f32>) -> Self {
        Self::from_quat(q)
    }
}

impl From<Rot3A> for GRot3<f32> {
    #[inline(always)]
    fn from(q: Rot3A) -> Self {
        q.to_quat()
    }
}

impl From<Rot3A> for crate::vector::GVec4<f32> {
    #[inline]
    fn from(q: Rot3A) -> Self {
        crate::vector::GVec4::new(q.x, q.y, q.z, q.w)
    }
}

impl From<Rot3A> for crate::vector::Vec4A {
    #[inline]
    fn from(q: Rot3A) -> Self {
        // Both are `repr(transparent)` over the same register in the same lane order.
        crate::vector::Vec4A::from_register(q.register())
    }
}

impl From<[f32; 4]> for Rot3A {
    #[inline]
    fn from(comps: [f32; 4]) -> Self {
        Self::from_array(comps)
    }
}

impl From<Rot3A> for [f32; 4] {
    #[inline]
    fn from(q: Rot3A) -> Self {
        q.to_array()
    }
}

impl From<(f32, f32, f32, f32)> for Rot3A {
    #[inline]
    fn from((x, y, z, w): (f32, f32, f32, f32)) -> Self {
        Self::from_xyzw(x, y, z, w)
    }
}

impl From<Rot3A> for (f32, f32, f32, f32) {
    #[inline]
    fn from(q: Rot3A) -> Self {
        (q.x, q.y, q.z, q.w)
    }
}
