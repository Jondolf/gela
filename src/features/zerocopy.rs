use gnum::num::Real;
use zerocopy::IntoBytes;

#[cfg(not(feature = "cuda"))]
use crate::isometry::GIso3;
use crate::{
    UnpaddedElement,
    affine::{GAffine2, GAffine3},
    isometry::GIso2,
    matrix::{GMat2, GMat3, GMat4},
    rotation::{GRot2, GRot3},
    vector::{GVec2, GVec3, GVec4},
};

// NOTE: Most of the traits are derived on the type definitions, but `IntoBytes`
//       is implemented here instead, because the `zerocopy` derive cannot prove
//       the absence of padding for a generic type with a fixed minimum alignment.

/// Implements [`IntoBytes`] for the given types.
macro_rules! impl_into_bytes {
    ($($ty:ident<$elem:ident: $bound:ident $(+ $extra:ident)*>),* $(,)?) => {
        $(
            // SAFETY: The type is `#[repr(C)]` with fields that are all `IntoBytes`, and its
            //         size is a multiple of its alignment, so it contains no padding bytes.
            unsafe impl<$elem: $bound + IntoBytes $(+ $extra)*> IntoBytes for $ty<$elem> {
                fn only_derive_is_allowed_to_implement_this_trait() {}
            }
        )*
    };
}

// The layout of these types is fully determined by their element type,
// so they never contain padding bytes.
impl_into_bytes!(
    GVec3<T: Copy>,
    GMat3<T: Real>,
    GAffine3<T: Real>,
);

// These types have a minimum alignment of 16 bytes, which only avoids padding
// if the size of the element type is a multiple of 4 bytes.
impl_into_bytes!(
    GVec4<T: Copy + UnpaddedElement>,
    GMat4<T: Real + UnpaddedElement>,
);

// With the `cuda` feature, `GVec2` and `GRot2` have a minimum alignment of 8 bytes and
// `GRot3` has a minimum alignment of 16 bytes, which the types built out of them inherit.
#[cfg(not(feature = "cuda"))]
impl_into_bytes!(
    GVec2<T: Copy>,
    GMat2<T: Real>,
    GRot2<T: Real>,
    GRot3<T: Real>,
    GAffine2<T: Real>,
    GIso2<T: Real>,
    GIso3<T: Real>,
);
#[cfg(feature = "cuda")]
impl_into_bytes!(
    GVec2<T: Copy + UnpaddedElement>,
    GMat2<T: Real + UnpaddedElement>,
    GRot2<T: Real + UnpaddedElement>,
    GRot3<T: Real + UnpaddedElement>,
    GAffine2<T: Real + UnpaddedElement>,
    GIso2<T: Real + UnpaddedElement>,
);

#[cfg(test)]
mod tests {
    use zerocopy::{FromBytes, FromZeros, IntoBytes};

    use crate::{
        affine::{Affine2, Affine3, DAffine2, DAffine3},
        isometry::{DIso2, DIso3, Iso2, Iso3},
        matrix::{DMat2, DMat3, DMat4, Mat2, Mat3, Mat4},
        rotation::{DRot2, DRot3, Rot2, Rot3},
        vector::{DVec2, DVec3, DVec4, IVec2, IVec3, IVec4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4},
    };

    /// Checks that a type can be read from bytes and written back as the same bytes.
    macro_rules! test_round_trip {
        ($name:ident, $ty:ident) => {
            #[test]
            fn $name() {
                let bytes = [0xAB_u8; size_of::<$ty>()];
                let value = <$ty>::read_from_bytes(&bytes).unwrap();

                // Viewing the value as bytes must produce the original bytes.
                assert_eq!(value.as_bytes(), &bytes[..]);
                assert_eq!(
                    value.as_bytes().as_ptr() as usize,
                    &value as *const $ty as usize
                );
            }
        };
    }

    /// Checks that a type can be read from bytes, but not written back as bytes.
    macro_rules! test_from_bytes {
        ($name:ident, $ty:ident) => {
            #[test]
            fn $name() {
                let bytes = [0_u8; size_of::<$ty>()];
                let value = <$ty>::read_from_bytes(&bytes).unwrap();
                assert_eq!(value, <$ty>::new_zeroed());
            }
        };
    }

    test_round_trip!(vec2, Vec2);
    test_round_trip!(vec3, Vec3);
    test_round_trip!(vec4, Vec4);
    test_round_trip!(dvec2, DVec2);
    test_round_trip!(dvec3, DVec3);
    test_round_trip!(dvec4, DVec4);
    test_round_trip!(ivec2, IVec2);
    test_round_trip!(ivec3, IVec3);
    test_round_trip!(ivec4, IVec4);
    test_round_trip!(uvec2, UVec2);
    test_round_trip!(uvec3, UVec3);
    test_round_trip!(uvec4, UVec4);

    test_round_trip!(mat2, Mat2);
    test_round_trip!(mat3, Mat3);
    test_round_trip!(mat4, Mat4);
    test_round_trip!(dmat2, DMat2);
    test_round_trip!(dmat3, DMat3);
    test_round_trip!(dmat4, DMat4);

    test_round_trip!(rot2, Rot2);
    test_round_trip!(rot3, Rot3);
    test_round_trip!(drot2, DRot2);
    test_round_trip!(drot3, DRot3);

    test_round_trip!(affine2, Affine2);
    test_round_trip!(affine3, Affine3);
    test_round_trip!(daffine2, DAffine2);
    test_round_trip!(daffine3, DAffine3);

    test_round_trip!(iso2, Iso2);
    test_round_trip!(diso2, DIso2);

    test_from_bytes!(iso3, Iso3);
    test_from_bytes!(diso3, DIso3);

    // Without the `cuda` feature, `GIso3` is free of padding and can also be viewed as bytes.
    #[cfg(not(feature = "cuda"))]
    test_round_trip!(iso3_round_trip, Iso3);
    #[cfg(not(feature = "cuda"))]
    test_round_trip!(diso3_round_trip, DIso3);
}
