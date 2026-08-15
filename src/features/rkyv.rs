#[cfg(feature = "cuda")]
use rkyv::munge::munge;
use rkyv::{
    Archive, Deserialize, Place, Portable, Serialize, rancor::Fallible,
    traits::CopyOptimization, traits::NoUndef,
};

#[cfg(feature = "cuda")]
use crate::isometry::GIso3;
use crate::{
    affine::{Affine2, Affine3, DAffine2, DAffine3},
    isometry::{DIso2, DIso3, Iso2, Iso3},
    matrix::{DMat2, DMat3, DMat4, Mat2, Mat3, Mat4},
    rotation::{DRot2, DRot3, Rot2, Rot3},
    vector::{DVec2, DVec3, DVec4, IVec2, IVec3, IVec4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4},
};

macro_rules! impl_rkyv_common {
    ($ty:ty) => {
        // SAFETY: The type is `#[repr(C)]` with a layout that only depends on its element type,
        //         and it has no interior mutability.
        unsafe impl Portable for $ty {}

        impl<S: Fallible + ?Sized> Serialize<S> for $ty {
            #[inline]
            fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
                Ok(())
            }
        }

        impl<D: Fallible + ?Sized> Deserialize<$ty, D> for $ty {
            #[inline]
            fn deserialize(&self, _: &mut D) -> Result<$ty, D::Error> {
                Ok(*self)
            }
        }

        #[cfg(feature = "bytecheck")]
        // SAFETY: Every bit pattern is valid for the element types of the type.
        unsafe impl<C: Fallible + ?Sized> rkyv::bytecheck::CheckBytes<C> for $ty {
            #[inline]
            unsafe fn check_bytes(_: *const Self, _: &mut C) -> Result<(), C::Error> {
                Ok(())
            }
        }
    };
}

macro_rules! impl_rkyv {
    ($($ty:ty: [$elem:ty; $count:literal]),* $(,)?) => {
        $(
            // Check that the type has no padding types.
            const _: () = assert!(size_of::<$ty>() == size_of::<$elem>() * $count);

            // SAFETY: The size of the type matches the size of its elements, as asserted above.
            unsafe impl NoUndef for $ty {}

            impl Archive for $ty {
                // SAFETY: The size of the type matches the size of its elements, as asserted above.
                const COPY_OPTIMIZATION: CopyOptimization<Self> =
                    unsafe { CopyOptimization::enable() };

                type Archived = $ty;
                type Resolver = ();

                #[inline]
                fn resolve(&self, _: Self::Resolver, out: Place<Self::Archived>) {
                    out.write(*self);
                }
            }

            impl_rkyv_common!($ty);
        )*
    };
}

#[cfg(feature = "cuda")]
macro_rules! impl_rkyv_padded {
    ($($ty:ty as $pattern:ident { $($field:ident),* $(,)? }),* $(,)?) => {
        $(
            impl Archive for $ty {
                type Archived = $ty;
                type Resolver = ();

                #[inline]
                fn resolve(&self, _: Self::Resolver, out: Place<Self::Archived>) {
                    // Write the fields one by one instead of writing the whole value,
                    // as a typed copy would mark the padding bytes as uninitialized.
                    munge!(let $pattern { $($field),* } = out);
                    $(
                        $field.write(self.$field);
                    )*
                }
            }

            impl_rkyv_common!($ty);
        )*
    };
}

impl_rkyv!(
    Vec2: [f32; 2],
    Vec3: [f32; 3],
    Vec4: [f32; 4],
    DVec2: [f64; 2],
    DVec3: [f64; 3],
    DVec4: [f64; 4],
    IVec2: [i32; 2],
    IVec3: [i32; 3],
    IVec4: [i32; 4],
    UVec2: [u32; 2],
    UVec3: [u32; 3],
    UVec4: [u32; 4],

    Mat2: [f32; 4],
    Mat3: [f32; 9],
    Mat4: [f32; 16],
    DMat2: [f64; 4],
    DMat3: [f64; 9],
    DMat4: [f64; 16],

    Rot2: [f32; 2],
    Rot3: [f32; 4],
    DRot2: [f64; 2],
    DRot3: [f64; 4],

    Affine2: [f32; 6],
    Affine3: [f32; 12],
    DAffine2: [f64; 6],
    DAffine3: [f64; 12],

    Iso2: [f32; 4],
    DIso2: [f64; 4],
);

#[cfg(not(feature = "cuda"))]
impl_rkyv!(
    Iso3: [f32; 7],
    DIso3: [f64; 7],
);

// With the `cuda` feature, `Iso3` and `DIso3` are padded by the 16-byte alignment
// of `Rot3` and `DRot3`.
#[cfg(feature = "cuda")]
impl_rkyv_padded!(
    Iso3 as GIso3 { rotation, translation },
    DIso3 as GIso3 { rotation, translation },
);

#[cfg(test)]
mod tests {
    use rkyv::rancor::Error;

    use crate::{
        affine::{Affine2, Affine3},
        isometry::{Iso2, Iso3},
        matrix::{Mat2, Mat3, Mat4},
        rotation::{Rot2, Rot3},
        vector::{IVec3, UVec2, Vec2, Vec3, Vec4},
    };

    /// Checks that a value survives an archiving round trip, and that the archived
    /// representation is the value itself.
    macro_rules! test_rkyv {
        ($name:ident, $ty:ident, $value:expr) => {
            #[test]
            fn $name() {
                let value: $ty = $value;

                let bytes = rkyv::to_bytes::<Error>(&value).unwrap();
                let archived = rkyv::access::<$ty, Error>(&bytes).unwrap();
                assert_eq!(archived, &value);

                let deserialized = rkyv::deserialize::<$ty, Error>(archived).unwrap();
                assert_eq!(deserialized, value);
            }
        };
    }

    test_rkyv!(vec2, Vec2, Vec2::new(1.0, 2.0));
    test_rkyv!(vec3, Vec3, Vec3::new(1.0, 2.0, 3.0));
    test_rkyv!(vec4, Vec4, Vec4::new(1.0, 2.0, 3.0, 4.0));
    test_rkyv!(ivec3, IVec3, IVec3::new(1, 2, 3));
    test_rkyv!(uvec2, UVec2, UVec2::new(1, 2));

    test_rkyv!(mat2, Mat2, Mat2::from_cols_array(&[1.0, 2.0, 3.0, 4.0]));
    test_rkyv!(mat3, Mat3, Mat3::IDENTITY);
    test_rkyv!(mat4, Mat4, Mat4::IDENTITY);

    test_rkyv!(rot2, Rot2, Rot2::IDENTITY);
    test_rkyv!(rot3, Rot3, Rot3::IDENTITY);

    test_rkyv!(affine3, Affine3, Affine3::IDENTITY);
    test_rkyv!(
        affine2,
        Affine2,
        Affine2::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    );

    test_rkyv!(
        iso2,
        Iso2,
        Iso2::from_rotation_translation(Rot2::IDENTITY, Vec2::new(1.0, 2.0))
    );
    test_rkyv!(
        iso3,
        Iso3,
        Iso3::from_rotation_translation(Rot3::IDENTITY, Vec3::new(1.0, 2.0, 3.0))
    );
}
