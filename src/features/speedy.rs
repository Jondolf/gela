use gnum::num::Real;
use speedy::{Context, Readable, Reader, Writable, Writer};

#[cfg(feature = "simd")]
use crate::{
    affine::{Affine2A, Affine3A},
    isometry::Iso3A,
    matrix::{Mat2A, Mat3A, Mat4A},
    rotation::Rot3A,
    vector::{BVec3A, BVec4A, Vec3A, Vec4A},
};
use crate::{
    affine::{GAffine2, GAffine3},
    isometry::{GIso2, GIso3},
    matrix::{GMat2, GMat3, GMat4},
    rotation::{GRot2, GRot3},
    vector::{GVec2, GVec3, GVec4},
};

macro_rules! impl_speedy {
    (
        $ty:ident<$elem:ident: $bound:ident>,
        $count:literal,
        |$value:ident| $to_array:expr,
        |$array:ident| $from_array:expr $(,)?
    ) => {
        impl<'a, C, $elem> Readable<'a, C> for $ty<$elem>
        where
            C: Context,
            $elem: $bound + Readable<'a, C>,
        {
            #[inline]
            fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
                let first = reader.read_value()?;

                let mut $array = [first; $count];
                for element in $array.iter_mut().skip(1) {
                    *element = reader.read_value()?;
                }

                Ok($from_array)
            }

            #[inline]
            fn minimum_bytes_needed() -> usize {
                <$elem as Readable<'a, C>>::minimum_bytes_needed() * $count
            }
        }

        impl<C: Context, $elem: $bound + Writable<C>> Writable<C> for $ty<$elem> {
            #[inline]
            fn write_to<W: ?Sized + Writer<C>>(&self, writer: &mut W) -> Result<(), C::Error> {
                let $value = self;
                let elements: [$elem; $count] = $to_array;

                for element in &elements {
                    writer.write_value(element)?;
                }

                Ok(())
            }

            #[inline]
            fn bytes_needed(&self) -> Result<usize, C::Error> {
                let $value = self;
                let elements: [$elem; $count] = $to_array;

                let mut size = 0;
                for element in &elements {
                    size += Writable::<C>::bytes_needed(element)?;
                }

                Ok(size)
            }
        }
    };
}

#[cfg(feature = "simd")]
macro_rules! impl_speedy_aligned {
    ($ty:ident, $elem:ty, $count:literal, |$value:ident| $to_array:expr, |$array:ident| $from_array:expr) => {
        impl<'a, C: Context> Readable<'a, C> for $ty
        where
            $elem: Readable<'a, C>,
        {
            fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
                let first = reader.read_value()?;
                let mut $array: [$elem; $count] = [first; $count];
                for element in $array.iter_mut().skip(1) {
                    *element = reader.read_value()?;
                }
                Ok($from_array)
            }
            fn minimum_bytes_needed() -> usize {
                <$elem as Readable<'a, C>>::minimum_bytes_needed() * $count
            }
        }
        impl<C: Context> Writable<C> for $ty
        where
            $elem: Writable<C>,
        {
            fn write_to<W: ?Sized + Writer<C>>(&self, writer: &mut W) -> Result<(), C::Error> {
                let $value = self;
                let elements: [$elem; $count] = $to_array;
                for element in &elements {
                    writer.write_value(element)?;
                }
                Ok(())
            }
            fn bytes_needed(&self) -> Result<usize, C::Error> {
                let $value = self;
                let elements: [$elem; $count] = $to_array;
                elements
                    .iter()
                    .try_fold(0, |size, e| Ok(size + Writable::<C>::bytes_needed(e)?))
            }
        }
    };
}

impl_speedy!(GVec2<T: Copy>, 2, |v| v.to_array(), |a| GVec2::from_array(a));
impl_speedy!(GVec3<T: Copy>, 3, |v| v.to_array(), |a| GVec3::from_array(a));
impl_speedy!(GVec4<T: Copy>, 4, |v| v.to_array(), |a| GVec4::from_array(a));

impl_speedy!(GMat2<T: Real>, 4, |m| m.to_cols_array(), |a| GMat2::from_cols_array(&a));
impl_speedy!(GMat3<T: Real>, 9, |m| m.to_cols_array(), |a| GMat3::from_cols_array(&a));
impl_speedy!(GMat4<T: Real>, 16, |m| m.to_cols_array(), |a| GMat4::from_cols_array(&a));

impl_speedy!(GRot2<T: Real>, 2, |r| r.to_array(), |a| GRot2::from_array(a));
impl_speedy!(GRot3<T: Real>, 4, |r| r.to_array(), |a| GRot3::from_array(a));

impl_speedy!(GAffine2<T: Real>, 6, |a| a.to_cols_array(), |a| GAffine2::from_cols_array(&a));
impl_speedy!(GAffine3<T: Real>, 12, |a| a.to_cols_array(), |a| GAffine3::from_cols_array(&a));

impl_speedy!(
    GIso2<T: Real>,
    4,
    |i| {
        let [cos, sin] = i.rotation.to_array();
        let [x, y] = i.translation.to_array();
        [cos, sin, x, y]
    },
    |a| GIso2::from_rotation_translation(
        GRot2::from_array([a[0], a[1]]),
        GVec2::from_array([a[2], a[3]]),
    ),
);

#[cfg(feature = "simd")]
impl_speedy_aligned!(BVec3A, bool, 3, |v| v.to_array(), |a| BVec3A::from_array(a));
#[cfg(feature = "simd")]
impl_speedy_aligned!(BVec4A, bool, 4, |v| v.to_array(), |a| BVec4A::from_array(a));
#[cfg(feature = "simd")]
impl_speedy_aligned!(Vec3A, f32, 3, |v| v.to_array(), |a| Vec3A::from_array(a));
#[cfg(feature = "simd")]
impl_speedy_aligned!(Vec4A, f32, 4, |v| v.to_array(), |a| Vec4A::from_array(a));
#[cfg(feature = "simd")]
impl_speedy_aligned!(Mat2A, f32, 4, |m| m.to_cols_array(), |a| {
    Mat2A::from_cols_array(&a)
});
#[cfg(feature = "simd")]
impl_speedy_aligned!(Mat3A, f32, 9, |m| m.to_cols_array(), |a| {
    Mat3A::from_cols_array(&a)
});
#[cfg(feature = "simd")]
impl_speedy_aligned!(Mat4A, f32, 16, |m| m.to_cols_array(), |a| {
    Mat4A::from_cols_array(&a)
});
#[cfg(feature = "simd")]
impl_speedy_aligned!(Rot3A, f32, 4, |r| r.to_array(), |a| Rot3A::from_array(a));
#[cfg(feature = "simd")]
impl_speedy_aligned!(Affine2A, f32, 6, |a| a.to_cols_array(), |a| {
    Affine2A::from_cols_array(&a)
});
#[cfg(feature = "simd")]
impl_speedy_aligned!(Affine3A, f32, 12, |a| a.to_cols_array(), |a| {
    Affine3A::from_cols_array(&a)
});
#[cfg(feature = "simd")]
impl_speedy_aligned!(
    Iso3A,
    f32,
    7,
    |i| {
        let [x, y, z, w] = i.rotation.to_array();
        let [tx, ty, tz] = i.translation.to_array();
        [x, y, z, w, tx, ty, tz]
    },
    |a| Iso3A::from_rotation_translation(
        Rot3A::from_array([a[0], a[1], a[2], a[3]]),
        Vec3A::from_array([a[4], a[5], a[6]]),
    )
);
impl_speedy!(
    GIso3<T: Real>,
    7,
    |i| {
        let [x, y, z, w] = i.rotation.to_array();
        let [tx, ty, tz] = i.translation.to_array();
        [x, y, z, w, tx, ty, tz]
    },
    |a| GIso3::from_rotation_translation(
        GRot3::from_array([a[0], a[1], a[2], a[3]]),
        GVec3::from_array([a[4], a[5], a[6]]),
    ),
);

#[cfg(test)]
mod tests {
    use speedy::{LittleEndian, Readable, Writable};

    use crate::{
        affine::{Affine2, Affine3},
        isometry::{Iso2, Iso3},
        matrix::{Mat2, Mat3, Mat4},
        rotation::{Rot2, Rot3},
        vector::{IVec3, Vec2, Vec3, Vec4},
    };

    /// Checks that a value survives a write and read round trip.
    macro_rules! test_speedy {
        ($name:ident, $ty:ident, $value:expr, $size:literal) => {
            #[test]
            fn $name() {
                let value: $ty = $value;

                let bytes = value.write_to_vec().unwrap();
                assert_eq!(bytes.len(), $size);
                assert_eq!(
                    Writable::<LittleEndian>::bytes_needed(&value).unwrap(),
                    $size
                );
                assert_eq!(
                    <$ty as Readable<'_, LittleEndian>>::minimum_bytes_needed(),
                    $size
                );

                assert_eq!(<$ty>::read_from_buffer(&bytes).unwrap(), value);
            }
        };
    }

    test_speedy!(vec2, Vec2, Vec2::new(1.0, 2.0), 8);
    test_speedy!(vec3, Vec3, Vec3::new(1.0, 2.0, 3.0), 12);
    test_speedy!(vec4, Vec4, Vec4::new(1.0, 2.0, 3.0, 4.0), 16);
    test_speedy!(ivec3, IVec3, IVec3::new(1, 2, 3), 12);

    test_speedy!(mat2, Mat2, Mat2::from_cols_array(&[1.0, 2.0, 3.0, 4.0]), 16);
    test_speedy!(mat3, Mat3, Mat3::IDENTITY, 36);
    test_speedy!(mat4, Mat4, Mat4::IDENTITY, 64);

    test_speedy!(rot2, Rot2, Rot2::IDENTITY, 8);
    test_speedy!(rot3, Rot3, Rot3::IDENTITY, 16);

    test_speedy!(affine2, Affine2, Affine2::IDENTITY, 24);
    test_speedy!(affine3, Affine3, Affine3::IDENTITY, 48);

    test_speedy!(
        iso2,
        Iso2,
        Iso2::from_rotation_translation(Rot2::IDENTITY, Vec2::new(1.0, 2.0)),
        16
    );
    test_speedy!(
        iso3,
        Iso3,
        Iso3::from_rotation_translation(Rot3::IDENTITY, Vec3::new(1.0, 2.0, 3.0)),
        28
    );
}
