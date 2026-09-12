use gnum::num::{Real, ScalarReal};
use mint::{
    ColumnMatrix2, ColumnMatrix2x3, ColumnMatrix3, ColumnMatrix3x4, ColumnMatrix4, EulerAngles,
    ExtraXYZ, ExtraZXZ, ExtraZYX, IntoMint, IntraXYZ, IntraZXZ, IntraZYX, Point2, Point3,
    Quaternion, RowMatrix2, RowMatrix2x3, RowMatrix3, RowMatrix3x4, RowMatrix4, Vector2, Vector3,
    Vector4,
};

use crate::{
    affine::{Affine2A, Affine3A, GAffine2, GAffine3},
    matrix::{GMat2, GMat3, GMat4, Mat2A, Mat3A, Mat4A},
    rotation::{EulerRot, GRot3, Rot3A},
    vector::{GVec2, GVec3, GVec4, Vec3A, Vec4A},
};

macro_rules! impl_vector {
    ($ty:ident, $mint:ident, [$($elem:ident),+ $(,)?]) => {
        impl<T: Copy> From<$mint<T>> for $ty<T> {
            #[inline]
            fn from(v: $mint<T>) -> Self {
                Self { $($elem: v.$elem),+ }
            }
        }

        impl<T: Copy> From<$ty<T>> for $mint<T> {
            #[inline]
            fn from(v: $ty<T>) -> Self {
                Self { $($elem: v.$elem),+ }
            }
        }
    };
}

impl_vector!(GVec2, Vector2, [x, y]);
impl_vector!(GVec2, Point2, [x, y]);
impl_vector!(GVec3, Vector3, [x, y, z]);
impl_vector!(GVec3, Point3, [x, y, z]);
impl_vector!(GVec4, Vector4, [x, y, z, w]);

impl<T: Copy> IntoMint for GVec2<T> {
    type MintType = Vector2<T>;
}

impl<T: Copy> IntoMint for GVec3<T> {
    type MintType = Vector3<T>;
}

impl<T: Copy> IntoMint for GVec4<T> {
    type MintType = Vector4<T>;
}

impl<T: Real> IntoMint for GRot3<T> {
    type MintType = Quaternion<T>;
}

impl<T: Real> From<Quaternion<T>> for GRot3<T> {
    #[inline]
    fn from(q: Quaternion<T>) -> Self {
        Self::from_xyzw(q.v.x, q.v.y, q.v.z, q.s)
    }
}

impl<T: Real> From<GRot3<T>> for Quaternion<T> {
    #[inline]
    fn from(q: GRot3<T>) -> Self {
        Self {
            v: Vector3 {
                x: q.x,
                y: q.y,
                z: q.z,
            },
            s: q.w,
        }
    }
}

macro_rules! impl_aligned_mint {
    ($aligned:ident, $generic:ident, $mint:ident) => {
        impl From<$mint<f32>> for $aligned {
            #[inline]
            fn from(value: $mint<f32>) -> Self {
                $generic::from(value).into()
            }
        }
        impl From<$aligned> for $mint<f32> {
            #[inline]
            fn from(value: $aligned) -> Self {
                $generic::from(value).into()
            }
        }
    };
}

impl_aligned_mint!(Vec3A, GVec3, Vector3);
impl_aligned_mint!(Vec3A, GVec3, Point3);
impl_aligned_mint!(Vec4A, GVec4, Vector4);
impl_aligned_mint!(Rot3A, GRot3, Quaternion);

impl IntoMint for Vec3A {
    type MintType = Vector3<f32>;
}
impl IntoMint for Vec4A {
    type MintType = Vector4<f32>;
}
impl IntoMint for Rot3A {
    type MintType = Quaternion<f32>;
}

macro_rules! impl_matrix {
    (
        $ty:ident as $value:ident,
        $column:ident { $($field:ident: $column_expr:expr),+ $(,)? },
        $row:ident,
        |$columns:ident| $from_columns:expr $(,)?
    ) => {
        impl<T: Real> IntoMint for $ty<T> {
            type MintType = $column<T>;
        }

        impl<T: Real> From<$column<T>> for $ty<T> {
            #[inline]
            fn from(m: $column<T>) -> Self {
                let $columns = [$(m.$field.into()),+];
                $from_columns
            }
        }

        impl<T: Real> From<$ty<T>> for $column<T> {
            #[inline]
            fn from($value: $ty<T>) -> Self {
                Self {
                    $($field: $column_expr.into(),)+
                }
            }
        }

        impl<T: Real> From<$row<T>> for $ty<T> {
            #[inline]
            fn from(m: $row<T>) -> Self {
                Self::from($column::from(m))
            }
        }

        impl<T: Real> From<$ty<T>> for $row<T> {
            #[inline]
            fn from(value: $ty<T>) -> Self {
                Self::from($column::from(value))
            }
        }
    };
}

impl_matrix!(
    GMat2 as m,
    ColumnMatrix2 {
        x: m.x_axis,
        y: m.y_axis,
    },
    RowMatrix2,
    |c| GMat2::from_cols(c[0], c[1]),
);

macro_rules! impl_aligned_matrix_mint {
    ($aligned:ident, $generic:ident, $column:ident, $row:ident) => {
        impl IntoMint for $aligned {
            type MintType = $column<f32>;
        }
        impl From<$column<f32>> for $aligned {
            #[inline]
            fn from(value: $column<f32>) -> Self {
                $generic::from(value).into()
            }
        }
        impl From<$aligned> for $column<f32> {
            #[inline]
            fn from(value: $aligned) -> Self {
                $generic::from(value).into()
            }
        }
        impl From<$row<f32>> for $aligned {
            #[inline]
            fn from(value: $row<f32>) -> Self {
                $generic::from(value).into()
            }
        }
        impl From<$aligned> for $row<f32> {
            #[inline]
            fn from(value: $aligned) -> Self {
                $generic::from(value).into()
            }
        }
    };
}

impl_aligned_matrix_mint!(Mat2A, GMat2, ColumnMatrix2, RowMatrix2);
impl_aligned_matrix_mint!(Mat3A, GMat3, ColumnMatrix3, RowMatrix3);
impl_aligned_matrix_mint!(Mat4A, GMat4, ColumnMatrix4, RowMatrix4);
impl_aligned_matrix_mint!(Affine2A, GAffine2, ColumnMatrix2x3, RowMatrix2x3);
impl_aligned_matrix_mint!(Affine3A, GAffine3, ColumnMatrix3x4, RowMatrix3x4);

impl_matrix!(
    GMat3 as m,
    ColumnMatrix3 {
        x: m.x_axis,
        y: m.y_axis,
        z: m.z_axis,
    },
    RowMatrix3,
    |c| GMat3::from_cols(c[0], c[1], c[2]),
);

impl_matrix!(
    GMat4 as m,
    ColumnMatrix4 {
        x: m.x_axis,
        y: m.y_axis,
        z: m.z_axis,
        w: m.w_axis,
    },
    RowMatrix4,
    |c| GMat4::from_cols(c[0], c[1], c[2], c[3]),
);

impl_matrix!(
    GAffine2 as a,
    ColumnMatrix2x3 {
        x: a.matrix2.x_axis,
        y: a.matrix2.y_axis,
        z: a.translation,
    },
    RowMatrix2x3,
    |c| GAffine2::from_cols(c[0], c[1], c[2]),
);

impl_matrix!(
    GAffine3 as a,
    ColumnMatrix3x4 {
        x: a.matrix3.x_axis,
        y: a.matrix3.y_axis,
        z: a.matrix3.z_axis,
        w: a.translation,
    },
    RowMatrix3x4,
    |c| GAffine3::from_cols(c[0], c[1], c[2], c[3]),
);

macro_rules! impl_euler_angles {
    ($($basis:ident => $order:ident),* $(,)?) => {
        $(
            impl<T: ScalarReal> From<EulerAngles<T, $basis>> for GRot3<T> {
                #[inline]
                fn from(angles: EulerAngles<T, $basis>) -> Self {
                    Self::from_euler(EulerRot::$order, angles.a, angles.b, angles.c)
                }
            }

            impl<T: ScalarReal> From<GRot3<T>> for EulerAngles<T, $basis> {
                #[inline]
                fn from(rotation: GRot3<T>) -> Self {
                    let (a, b, c) = rotation.to_euler(EulerRot::$order);
                    Self::from([a, b, c])
                }
            }
        )*
    };
}

impl_euler_angles!(
    IntraXYZ => XYZ,
    IntraZXZ => ZXZ,
    IntraZYX => ZYX,
    ExtraXYZ => XYZEx,
    ExtraZXZ => ZXZEx,
    ExtraZYX => ZYXEx,
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        affine::{Affine2, Affine3},
        matrix::{Mat2, Mat3, Mat4},
        rotation::Rot3,
        vector::{Vec2, Vec3, Vec4},
    };

    #[test]
    fn vectors() {
        let vector = Vec3::new(1.0, 2.0, 3.0);

        assert_eq!(Vector3::from(vector), Vector3::from([1.0, 2.0, 3.0]));
        assert_eq!(Vec3::from(Vector3::from(vector)), vector);

        assert_eq!(Point3::from(vector), Point3::from([1.0, 2.0, 3.0]));
        assert_eq!(Vec3::from(Point3::from(vector)), vector);

        assert_eq!(
            Vec2::from(Point2::from(Vec2::new(1.0, 2.0))),
            Vec2::new(1.0, 2.0)
        );
        assert_eq!(
            Vec4::from(Vector4::from(Vec4::new(1.0, 2.0, 3.0, 4.0))),
            Vec4::new(1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn quaternion() {
        let rotation = Rot3::from_xyzw(1.0, 2.0, 3.0, 4.0);
        let quaternion = Quaternion::from(rotation);

        assert_eq!(quaternion.v, Vector3::from([1.0, 2.0, 3.0]));
        assert_eq!(quaternion.s, 4.0);
        assert_eq!(Rot3::from(quaternion), rotation);
    }

    #[test]
    fn matrices() {
        let matrix = Mat3::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let columns = ColumnMatrix3::from(matrix);
        assert_eq!(columns.x, Vector3::from([1.0, 2.0, 3.0]));
        assert_eq!(columns.z, Vector3::from([7.0, 8.0, 9.0]));
        assert_eq!(Mat3::from(columns), matrix);

        let rows = RowMatrix3::from(matrix);
        assert_eq!(rows.x, Vector3::from([1.0, 4.0, 7.0]));
        assert_eq!(rows.z, Vector3::from([3.0, 6.0, 9.0]));
        assert_eq!(Mat3::from(rows), matrix);

        let matrix = Mat2::from_cols_array(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Mat2::from(ColumnMatrix2::from(matrix)), matrix);
        assert_eq!(Mat2::from(RowMatrix2::from(matrix)), matrix);
        assert_eq!(RowMatrix2::from(matrix).x, Vector2::from([1.0, 3.0]));

        let matrix = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        assert_eq!(Mat4::from(ColumnMatrix4::from(matrix)), matrix);
        assert_eq!(Mat4::from(RowMatrix4::from(matrix)), matrix);
        assert_eq!(
            RowMatrix4::from(matrix).x,
            Vector4::from([1.0, 5.0, 9.0, 13.0])
        );
    }

    #[test]
    fn affine_transformations() {
        let affine = Affine2::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let columns = ColumnMatrix2x3::from(affine);
        assert_eq!(columns.x, Vector2::from([1.0, 2.0]));
        assert_eq!(columns.z, Vector2::from([5.0, 6.0]));
        assert_eq!(Affine2::from(columns), affine);

        let rows = RowMatrix2x3::from(affine);
        assert_eq!(rows.x, Vector3::from([1.0, 3.0, 5.0]));
        assert_eq!(rows.y, Vector3::from([2.0, 4.0, 6.0]));
        assert_eq!(Affine2::from(rows), affine);

        let affine = Affine3::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        assert_eq!(Affine3::from(ColumnMatrix3x4::from(affine)), affine);
        assert_eq!(Affine3::from(RowMatrix3x4::from(affine)), affine);
        assert_eq!(
            RowMatrix3x4::from(affine).x,
            Vector4::from([1.0, 4.0, 7.0, 10.0])
        );
    }

    #[test]
    fn euler_angles() {
        let angles = EulerAngles::<f32, IntraXYZ>::from([0.5, 0.25, 0.125]);
        let rotation = Rot3::from(angles);

        assert!(rotation.abs_diff_eq(Rot3::from_euler(EulerRot::XYZ, 0.5, 0.25, 0.125), 1e-6));

        let round_trip = EulerAngles::<f32, IntraXYZ>::from(rotation);
        assert!((round_trip.a - angles.a).abs() < 1e-6);
        assert!((round_trip.b - angles.b).abs() < 1e-6);
        assert!((round_trip.c - angles.c).abs() < 1e-6);

        let extrinsic = EulerAngles::<f32, ExtraXYZ>::from([0.5, 0.25, 0.125]);
        assert!(
            Rot3::from(extrinsic)
                .abs_diff_eq(Rot3::from_euler(EulerRot::XYZEx, 0.5, 0.25, 0.125), 1e-6)
        );
    }

    #[test]
    fn isometries_convert_through_affine_transformations() {
        use crate::{affine::Affine3, isometry::Iso3};

        let isometry = Iso3::from_rotation_translation(
            Rot3::from_axis_angle(Vec3::Z, 0.5),
            Vec3::new(1.0, 2.0, 3.0),
        );

        let affine = Affine3::from_rotation_translation(isometry.rotation, isometry.translation);
        let columns = ColumnMatrix3x4::from(affine);

        assert_eq!(columns.w, Vector3::from([1.0, 2.0, 3.0]));
        assert_eq!(Affine3::from(columns), affine);
    }

    #[test]
    fn into_mint_types() {
        fn assert_mint_type<T: IntoMint<MintType = M>, M>() {}

        assert_mint_type::<Vec2, Vector2<f32>>();
        assert_mint_type::<Vec3, Vector3<f32>>();
        assert_mint_type::<Vec4, Vector4<f32>>();
        assert_mint_type::<Mat2, ColumnMatrix2<f32>>();
        assert_mint_type::<Mat3, ColumnMatrix3<f32>>();
        assert_mint_type::<Mat4, ColumnMatrix4<f32>>();
        assert_mint_type::<Rot3, Quaternion<f32>>();
        assert_mint_type::<Affine2, ColumnMatrix2x3<f32>>();
        assert_mint_type::<Affine3, ColumnMatrix3x4<f32>>();
    }
}
