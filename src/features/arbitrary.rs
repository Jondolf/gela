use arbitrary::{Arbitrary, Result, Unstructured};
use gnum::num::Real;

use crate::{
    affine::{GAffine2, GAffine3},
    isometry::{GIso2, GIso3},
    matrix::{GMat2, GMat3, GMat4},
    rotation::{GRot2, GRot3},
    vector::{GVec2, GVec3, GVec4},
};

macro_rules! impl_arbitrary {
    (
        $ty:ident<$elem:ident: $bound:ident>,
        $count:literal,
        |$array:ident| $from_array:expr $(,)?
    ) => {
        impl<'a, $elem: $bound + Arbitrary<'a>> Arbitrary<'a> for $ty<$elem> {
            fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
                let $array: [$elem; $count] = u.arbitrary()?;
                Ok($from_array)
            }

            fn size_hint(depth: usize) -> (usize, Option<usize>) {
                <[$elem; $count] as Arbitrary<'a>>::size_hint(depth)
            }
        }
    };
}

impl_arbitrary!(GVec2<T: Copy>, 2, |a| GVec2::from_array(a));
impl_arbitrary!(GVec3<T: Copy>, 3, |a| GVec3::from_array(a));
impl_arbitrary!(GVec4<T: Copy>, 4, |a| GVec4::from_array(a));

impl_arbitrary!(GMat2<T: Real>, 4, |a| GMat2::from_cols_array(&a));
impl_arbitrary!(GMat3<T: Real>, 9, |a| GMat3::from_cols_array(&a));
impl_arbitrary!(GMat4<T: Real>, 16, |a| GMat4::from_cols_array(&a));

impl_arbitrary!(GRot2<T: Real>, 2, |a| GRot2::from_array(a));
impl_arbitrary!(GRot3<T: Real>, 4, |a| GRot3::from_array(a));

impl_arbitrary!(GAffine2<T: Real>, 6, |a| GAffine2::from_cols_array(&a));
impl_arbitrary!(GAffine3<T: Real>, 12, |a| GAffine3::from_cols_array(&a));

impl_arbitrary!(GIso2<T: Real>, 4, |a| GIso2::from_rotation_translation(
    GRot2::from_array([a[0], a[1]]),
    GVec2::from_array([a[2], a[3]]),
));
impl_arbitrary!(GIso3<T: Real>, 7, |a| GIso3::from_rotation_translation(
    GRot3::from_array([a[0], a[1], a[2], a[3]]),
    GVec3::from_array([a[4], a[5], a[6]]),
));

#[cfg(test)]
mod tests {
    use arbitrary::{Arbitrary, Unstructured};

    use crate::{
        affine::{Affine2, Affine3},
        isometry::{Iso2, Iso3},
        matrix::{Mat2, Mat3, Mat4},
        rotation::{Rot2, Rot3},
        vector::{IVec3, Vec2, Vec3, Vec4},
    };

    fn f32_bytes() -> [u8; 16 * size_of::<f32>()] {
        let mut bytes = [0_u8; 16 * size_of::<f32>()];
        for (index, chunk) in bytes.chunks_exact_mut(size_of::<f32>()).enumerate() {
            chunk.copy_from_slice(&((index + 1) as f32).to_le_bytes());
        }
        bytes
    }

    /// Checks that a type is built from arbitrary elements in the expected order.
    macro_rules! test_arbitrary {
        ($name:ident, $expected:expr) => {
            #[test]
            fn $name() {
                let bytes = f32_bytes();
                let mut unstructured = Unstructured::new(&bytes);
                assert_eq!($expected, unstructured.arbitrary().unwrap());
            }
        };
    }

    test_arbitrary!(vec2, Vec2::new(1.0, 2.0));
    test_arbitrary!(vec3, Vec3::new(1.0, 2.0, 3.0));
    test_arbitrary!(vec4, Vec4::new(1.0, 2.0, 3.0, 4.0));

    test_arbitrary!(mat2, Mat2::from_cols_array(&[1.0, 2.0, 3.0, 4.0]));
    test_arbitrary!(
        mat3,
        Mat3::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    );
    test_arbitrary!(
        mat4,
        Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
        ])
    );

    test_arbitrary!(rot2, Rot2::from_array([1.0, 2.0]));
    test_arbitrary!(rot3, Rot3::from_array([1.0, 2.0, 3.0, 4.0]));

    test_arbitrary!(
        affine2,
        Affine2::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    );
    test_arbitrary!(
        affine3,
        Affine3::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ])
    );

    test_arbitrary!(
        iso2,
        Iso2::from_rotation_translation(Rot2::from_array([1.0, 2.0]), Vec2::new(3.0, 4.0))
    );
    test_arbitrary!(
        iso3,
        Iso3::from_rotation_translation(
            Rot3::from_array([1.0, 2.0, 3.0, 4.0]),
            Vec3::new(5.0, 6.0, 7.0)
        )
    );

    #[test]
    fn integer_vector() {
        let bytes = [1_u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
        let mut unstructured = Unstructured::new(&bytes);
        assert_eq!(
            IVec3::new(1, 2, 3),
            IVec3::arbitrary(&mut unstructured).unwrap()
        );
    }

    #[test]
    fn size_hint() {
        assert_eq!(
            <Vec3 as Arbitrary<'_>>::size_hint(0),
            <[f32; 3] as Arbitrary<'_>>::size_hint(0)
        );
    }
}
