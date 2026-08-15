#[cfg(feature = "cuda")]
use bytemuck::AnyBitPattern;
use bytemuck::{Pod, Zeroable};
use gnum::num::Real;

use crate::{
    UnpaddedElement,
    affine::{GAffine2, GAffine3},
    isometry::{GIso2, GIso3},
    matrix::{GMat2, GMat3, GMat4},
    rotation::{GRot2, GRot3},
    vector::{GVec2, GVec3, GVec4},
};

macro_rules! impl_zeroable {
    ($($ty:ident<$elem:ident: $bound:ident>),* $(,)?) => {
        $(
            // SAFETY: An all-zero bit pattern is valid for the type if it is valid
            //         for every element, and zeroing any padding bytes is always allowed.
            unsafe impl<$elem: $bound + Zeroable> Zeroable for $ty<$elem> {}
        )*
    };
}

macro_rules! impl_pod {
    ($($ty:ident<$elem:ident: $bound:ident $(+ $extra:ident)*>),* $(,)?) => {
        $(
            // SAFETY: The type is `#[repr(C)]` with fields that are all `Pod`, and its size
            //         is a multiple of its alignment, so it contains no padding bytes
            //         and every bit pattern is valid for it.
            unsafe impl<$elem: $bound + Pod $(+ $extra)*> Pod for $ty<$elem> {}
        )*
    };
}

#[cfg(feature = "cuda")]
macro_rules! impl_any_bit_pattern {
    ($($ty:ident<$elem:ident: $bound:ident>),* $(,)?) => {
        $(
            // SAFETY: Every bit pattern is valid for the type if it is valid for every element.
            //         Padding bytes are allowed by `AnyBitPattern`, as it does not permit
            //         reading the type as bytes.
            unsafe impl<$elem: $bound + AnyBitPattern> AnyBitPattern for $ty<$elem> {}
        )*
    };
}

impl_zeroable!(
    GVec2<T: Copy>,
    GVec3<T: Copy>,
    GVec4<T: Copy>,
    GMat2<T: Real>,
    GMat3<T: Real>,
    GMat4<T: Real>,
    GRot2<T: Real>,
    GRot3<T: Real>,
    GAffine2<T: Real>,
    GAffine3<T: Real>,
    GIso2<T: Real>,
    GIso3<T: Real>,
);

// The layout of these types is fully determined by their element type,
// so they never contain padding bytes.
impl_pod!(
    GVec3<T: Copy>,
    GMat3<T: Real>,
    GAffine3<T: Real>,
);

// These types have a minimum alignment of 16 bytes, which only avoids padding
// if the size of the element type is a multiple of 4 bytes.
impl_pod!(
    GVec4<T: Copy + UnpaddedElement>,
    GMat4<T: Real + UnpaddedElement>,
);

// With the `cuda` feature, `GVec2` and `GRot2` have a minimum alignment of 8 bytes and
// `GRot3` has a minimum alignment of 16 bytes, which the types built out of them inherit.
#[cfg(not(feature = "cuda"))]
impl_pod!(
    GVec2<T: Copy>,
    GMat2<T: Real>,
    GRot2<T: Real>,
    GRot3<T: Real>,
    GAffine2<T: Real>,
    GIso2<T: Real>,
    GIso3<T: Real>,
);
#[cfg(feature = "cuda")]
impl_pod!(
    GVec2<T: Copy + UnpaddedElement>,
    GMat2<T: Real + UnpaddedElement>,
    GRot2<T: Real + UnpaddedElement>,
    GRot3<T: Real + UnpaddedElement>,
    GAffine2<T: Real + UnpaddedElement>,
    GIso2<T: Real + UnpaddedElement>,
);

// With the `cuda` feature, the 16-byte alignment of `GRot3` leaves padding bytes in `GIso3`
// for most element types, so it can only be read from bytes, not viewed as bytes.
#[cfg(feature = "cuda")]
impl_any_bit_pattern!(
    GIso3<T: Real>,
);

#[cfg(test)]
mod tests {
    use crate::{
        affine::{Affine2, Affine3, DAffine2, DAffine3},
        isometry::{DIso2, DIso3, Iso2, Iso3},
        matrix::{DMat2, DMat3, DMat4, Mat2, Mat3, Mat4},
        rotation::{DRot2, DRot3, Rot2, Rot3},
        vector::{DVec2, DVec3, DVec4, IVec2, IVec3, IVec4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4},
    };

    /// Checks that a `Pod` type can be created from arbitrary bytes
    /// and viewed as the same bytes again.
    macro_rules! test_pod {
        ($name:ident, $ty:ident) => {
            #[test]
            fn $name() {
                let bytes = [0xAB_u8; size_of::<$ty>()];
                let value = bytemuck::pod_read_unaligned::<$ty>(&bytes);

                // Viewing the value as bytes must produce the original bytes.
                assert_eq!(bytemuck::bytes_of(&value), &bytes[..]);
                assert_eq!(
                    bytemuck::bytes_of(&value).as_ptr() as usize,
                    &value as *const $ty as usize
                );
            }
        };
    }

    /// Checks that an `AnyBitPattern` type can be created from arbitrary bytes.
    macro_rules! test_any_bit_pattern {
        ($name:ident, $ty:ident) => {
            #[test]
            fn $name() {
                let bytes = [0_u8; size_of::<$ty>()];
                let value = bytemuck::pod_read_unaligned::<$ty>(&bytes);
                assert_eq!(value, bytemuck::Zeroable::zeroed());
            }
        };
    }

    test_pod!(vec2, Vec2);
    test_pod!(vec3, Vec3);
    test_pod!(vec4, Vec4);
    test_pod!(dvec2, DVec2);
    test_pod!(dvec3, DVec3);
    test_pod!(dvec4, DVec4);
    test_pod!(ivec2, IVec2);
    test_pod!(ivec3, IVec3);
    test_pod!(ivec4, IVec4);
    test_pod!(uvec2, UVec2);
    test_pod!(uvec3, UVec3);
    test_pod!(uvec4, UVec4);

    test_pod!(mat2, Mat2);
    test_pod!(mat3, Mat3);
    test_pod!(mat4, Mat4);
    test_pod!(dmat2, DMat2);
    test_pod!(dmat3, DMat3);
    test_pod!(dmat4, DMat4);

    test_pod!(rot2, Rot2);
    test_pod!(rot3, Rot3);
    test_pod!(drot2, DRot2);
    test_pod!(drot3, DRot3);

    test_pod!(affine2, Affine2);
    test_pod!(affine3, Affine3);
    test_pod!(daffine2, DAffine2);
    test_pod!(daffine3, DAffine3);

    test_pod!(iso2, Iso2);
    test_pod!(diso2, DIso2);

    test_any_bit_pattern!(iso3, Iso3);
    test_any_bit_pattern!(diso3, DIso3);

    // Without the `cuda` feature, `GIso3` is free of padding and can also be viewed as bytes.
    #[cfg(not(feature = "cuda"))]
    test_pod!(iso3_pod, Iso3);
    #[cfg(not(feature = "cuda"))]
    test_pod!(diso3_pod, DIso3);

    #[cfg(feature = "portable_simd")]
    #[test]
    fn simd_vec4() {
        use crate::vector::GVec4;

        type Vec4x4 = GVec4<gnum::f32x4>;

        let bytes = [0xAB_u8; size_of::<Vec4x4>()];
        let value = bytemuck::pod_read_unaligned::<Vec4x4>(&bytes);

        assert_eq!(bytemuck::bytes_of(&value), &bytes[..]);
    }
}
