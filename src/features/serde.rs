use core::{fmt, marker::PhantomData};

use gnum::num::Real;
use serde_core::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{EnumAccess, Error, SeqAccess, Unexpected, VariantAccess, Visitor},
    ser::SerializeTupleStruct,
};

use crate::{
    affine::{GAffine2, GAffine3},
    isometry::{GIso2, GIso3},
    matrix::{GMat2, GMat3, GMat4},
    rotation::{EulerRot, GRot2, GRot3},
    vector::{GVec2, GVec3, GVec4},
};

macro_rules! impl_serde {
    (
        $ty:ident<$elem:ident: $bound:ident>,
        $count:literal,
        |$value:ident| $to_array:expr,
        |$array:ident| $from_array:expr $(,)?
    ) => {
        #[doc = concat!("Serializes as a sequence of ", stringify!($count), " elements.")]
        impl<$elem: $bound + Serialize> Serialize for $ty<$elem> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let $value = self;
                let elements: [$elem; $count] = $to_array;

                let mut state = serializer.serialize_tuple_struct(stringify!($ty), $count)?;
                for element in &elements {
                    state.serialize_field(element)?;
                }
                state.end()
            }
        }

        #[doc = concat!("Deserializes from a sequence of ", stringify!($count), " elements.")]
        impl<'de, $elem: $bound + Deserialize<'de>> Deserialize<'de> for $ty<$elem> {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct SeqVisitor<$elem>(PhantomData<$elem>);

                impl<'de, $elem: $bound + Deserialize<'de>> Visitor<'de> for SeqVisitor<$elem> {
                    type Value = $ty<$elem>;

                    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                        formatter.write_str(concat!(
                            "a sequence of ",
                            stringify!($count),
                            " ",
                            stringify!($ty),
                            " elements"
                        ))
                    }

                    fn visit_seq<A: SeqAccess<'de>>(
                        self,
                        mut seq: A,
                    ) -> Result<Self::Value, A::Error> {
                        let first = seq
                            .next_element()?
                            .ok_or_else(|| Error::invalid_length(0, &self))?;

                        let mut $array = [first; $count];
                        for index in 1..$count {
                            $array[index] = seq
                                .next_element()?
                                .ok_or_else(|| Error::invalid_length(index, &self))?;
                        }

                        Ok($from_array)
                    }
                }

                deserializer.deserialize_tuple_struct(
                    stringify!($ty),
                    $count,
                    SeqVisitor(PhantomData),
                )
            }
        }
    };
}

impl_serde!(GVec2<T: Copy>, 2, |v| v.to_array(), |a| GVec2::from_array(a));
impl_serde!(GVec3<T: Copy>, 3, |v| v.to_array(), |a| GVec3::from_array(a));
impl_serde!(GVec4<T: Copy>, 4, |v| v.to_array(), |a| GVec4::from_array(a));

impl_serde!(GMat2<T: Real>, 4, |m| m.to_cols_array(), |a| GMat2::from_cols_array(&a));
impl_serde!(GMat3<T: Real>, 9, |m| m.to_cols_array(), |a| GMat3::from_cols_array(&a));
impl_serde!(GMat4<T: Real>, 16, |m| m.to_cols_array(), |a| GMat4::from_cols_array(&a));

impl_serde!(GRot2<T: Real>, 2, |r| r.to_array(), |a| GRot2::from_array(a));
impl_serde!(GRot3<T: Real>, 4, |r| r.to_array(), |a| GRot3::from_array(a));

impl_serde!(GAffine2<T: Real>, 6, |a| a.to_cols_array(), |a| GAffine2::from_cols_array(&a));
impl_serde!(GAffine3<T: Real>, 12, |a| a.to_cols_array(), |a| GAffine3::from_cols_array(&a));

impl_serde!(
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
impl_serde!(
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

macro_rules! impl_serde_unit_enum {
    ($ty:ident, [$($variant:ident),* $(,)?]) => {
        /// The names of the variants, in declaration order.
        const VARIANT_NAMES: &[&str] = &[$(stringify!($variant)),*];

        /// The variants themselves, in declaration order.
        const VARIANT_VALUES: &[$ty] = &[$($ty::$variant),*];

        // The variants must be listed in declaration order, as the discriminant of a variant
        // is used to index into the tables above.
        const _: () = {
            let mut index = 0;
            while index < VARIANT_VALUES.len() {
                assert!(VARIANT_VALUES[index] as u32 == index as u32);
                index += 1;
            }
        };

        impl Serialize for $ty {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                // The enum is fieldless with implicit discriminants,
                // so the discriminant is the index of the variant.
                let index = *self as u32;
                serializer.serialize_unit_variant(
                    stringify!($ty),
                    index,
                    VARIANT_NAMES[index as usize],
                )
            }
        }

        impl<'de> Deserialize<'de> for $ty {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct EnumVisitor;

                impl<'de> Visitor<'de> for EnumVisitor {
                    type Value = $ty;

                    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                        formatter.write_str(concat!("a ", stringify!($ty), " variant"))
                    }

                    fn visit_enum<A: EnumAccess<'de>>(
                        self,
                        data: A,
                    ) -> Result<Self::Value, A::Error> {
                        let (value, variant) = data.variant::<Identifier>()?;
                        variant.unit_variant()?;
                        Ok(value.0)
                    }
                }

                /// The identifier of a variant, deserialized from a name or an index.
                struct Identifier($ty);

                impl<'de> Deserialize<'de> for Identifier {
                    fn deserialize<D: Deserializer<'de>>(
                        deserializer: D,
                    ) -> Result<Self, D::Error> {
                        struct IdentifierVisitor;

                        impl<'de> Visitor<'de> for IdentifierVisitor {
                            type Value = Identifier;

                            fn expecting(
                                &self,
                                formatter: &mut fmt::Formatter<'_>,
                            ) -> fmt::Result {
                                formatter.write_str("a variant identifier")
                            }

                            fn visit_u64<E: Error>(self, value: u64) -> Result<Self::Value, E> {
                                usize::try_from(value)
                                    .ok()
                                    .and_then(|index| VARIANT_VALUES.get(index))
                                    .map(|&variant| Identifier(variant))
                                    .ok_or_else(|| {
                                        Error::invalid_value(
                                            Unexpected::Unsigned(value),
                                            &"a variant index",
                                        )
                                    })
                            }

                            fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
                                match value {
                                    $(stringify!($variant) => Ok(Identifier($ty::$variant)),)*
                                    _ => Err(Error::unknown_variant(value, VARIANT_NAMES)),
                                }
                            }
                        }

                        deserializer.deserialize_identifier(IdentifierVisitor)
                    }
                }

                deserializer.deserialize_enum(stringify!($ty), VARIANT_NAMES, EnumVisitor)
            }
        }
    };
}

impl_serde_unit_enum!(
    EulerRot,
    [
        ZYX, ZXY, YXZ, YZX, XYZ, XZY, ZYZ, ZXZ, YXY, YZY, XYX, XZX, ZYXEx, ZXYEx, YXZEx, YZXEx,
        XYZEx, XZYEx, ZYZEx, ZXZEx, YXYEx, YZYEx, XYXEx, XZXEx,
    ]
);

#[cfg(test)]
mod tests {
    use crate::{
        affine::{Affine2, Affine3},
        isometry::{Iso2, Iso3},
        matrix::{Mat2, Mat3, Mat4},
        rotation::{Rot2, Rot3},
        vector::{IVec3, Vec2, Vec3, Vec4},
    };

    /// Checks that a value is serialized as the expected JSON, and deserializes back into it.
    macro_rules! test_serde {
        ($name:ident, $value:expr, $json:literal) => {
            #[test]
            fn $name() {
                let value = $value;

                let serialized = serde_json::to_string(&value).unwrap();
                assert_eq!(serialized, $json);

                let deserialized = serde_json::from_str(&serialized).unwrap();
                assert_eq!(value, deserialized);
            }
        };
    }

    test_serde!(vec2, Vec2::new(1.0, 2.0), "[1.0,2.0]");
    test_serde!(vec3, Vec3::new(1.0, 2.0, 3.0), "[1.0,2.0,3.0]");
    test_serde!(vec4, Vec4::new(1.0, 2.0, 3.0, 4.0), "[1.0,2.0,3.0,4.0]");
    test_serde!(ivec3, IVec3::new(1, 2, 3), "[1,2,3]");

    test_serde!(
        mat2,
        Mat2::from_cols_array(&[1.0, 2.0, 3.0, 4.0]),
        "[1.0,2.0,3.0,4.0]"
    );
    test_serde!(
        mat3,
        Mat3::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        "[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]"
    );
    test_serde!(
        mat4,
        Mat4::IDENTITY,
        "[1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]"
    );

    test_serde!(rot2, Rot2::IDENTITY, "[1.0,0.0]");
    test_serde!(rot3, Rot3::IDENTITY, "[0.0,0.0,0.0,1.0]");

    test_serde!(
        affine2,
        Affine2::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "[1.0,2.0,3.0,4.0,5.0,6.0]"
    );
    test_serde!(
        affine3,
        Affine3::IDENTITY,
        "[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]"
    );

    test_serde!(
        iso2,
        Iso2::from_rotation_translation(Rot2::IDENTITY, Vec2::new(1.0, 2.0)),
        "[1.0,0.0,1.0,2.0]"
    );
    test_serde!(
        iso3,
        Iso3::from_rotation_translation(Rot3::IDENTITY, Vec3::new(1.0, 2.0, 3.0)),
        "[0.0,0.0,0.0,1.0,1.0,2.0,3.0]"
    );

    #[test]
    fn euler_rot() {
        use crate::rotation::EulerRot;

        for variant in super::VARIANT_VALUES {
            let serialized = serde_json::to_string(variant).unwrap();
            let deserialized: EulerRot = serde_json::from_str(&serialized).unwrap();
            assert_eq!(*variant, deserialized);
        }

        assert_eq!(
            serde_json::to_string(&EulerRot::XYZEx).unwrap(),
            "\"XYZEx\""
        );
        assert_eq!(
            serde_json::from_str::<EulerRot>("\"ZYX\"").unwrap(),
            EulerRot::ZYX
        );
        assert!(serde_json::from_str::<EulerRot>("\"NotAVariant\"").is_err());
    }

    #[test]
    fn invalid_length() {
        assert!(serde_json::from_str::<Vec3>("[1.0,2.0]").is_err());
        assert!(serde_json::from_str::<Vec3>("[1.0,2.0,3.0,4.0]").is_err());
        assert!(serde_json::from_str::<Vec3>("[]").is_err());
    }
}
