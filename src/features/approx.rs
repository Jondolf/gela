use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use gnum::num::Real;

use crate::{
    affine::{Affine2A, Affine3A, GAffine2, GAffine3},
    isometry::{GIso2, GIso3, Iso3A},
    matrix::{GMat2, GMat3, GMat4, Mat2A, Mat3A, Mat4A},
    rotation::{GRot2, GRot3, Rot3A},
    vector::{GVec2, GVec3, GVec4, Vec3A, Vec4A},
};

macro_rules! impl_approx {
    (
        $ty:ident<$elem:ident: $bound:ident>,
        $count:literal,
        |$value:ident| $to_array:expr $(,)?
    ) => {
        impl<$elem: $bound + AbsDiffEq> AbsDiffEq for $ty<$elem>
        where
            $elem::Epsilon: Clone,
        {
            type Epsilon = $elem::Epsilon;

            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                $elem::default_epsilon()
            }

            #[inline]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                let $value = self;
                let elements: [$elem; $count] = $to_array;
                let $value = other;
                let other_elements: [$elem; $count] = $to_array;

                elements
                    .iter()
                    .zip(other_elements.iter())
                    .all(|(a, b)| AbsDiffEq::abs_diff_eq(a, b, epsilon.clone()))
            }
        }

        impl<$elem: $bound + RelativeEq> RelativeEq for $ty<$elem>
        where
            $elem::Epsilon: Clone,
        {
            #[inline]
            fn default_max_relative() -> Self::Epsilon {
                $elem::default_max_relative()
            }

            #[inline]
            fn relative_eq(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                let $value = self;
                let elements: [$elem; $count] = $to_array;
                let $value = other;
                let other_elements: [$elem; $count] = $to_array;

                elements.iter().zip(other_elements.iter()).all(|(a, b)| {
                    RelativeEq::relative_eq(a, b, epsilon.clone(), max_relative.clone())
                })
            }
        }

        impl<$elem: $bound + UlpsEq> UlpsEq for $ty<$elem>
        where
            $elem::Epsilon: Clone,
        {
            #[inline]
            fn default_max_ulps() -> u32 {
                $elem::default_max_ulps()
            }

            #[inline]
            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                let $value = self;
                let elements: [$elem; $count] = $to_array;
                let $value = other;
                let other_elements: [$elem; $count] = $to_array;

                elements
                    .iter()
                    .zip(other_elements.iter())
                    .all(|(a, b)| UlpsEq::ulps_eq(a, b, epsilon.clone(), max_ulps))
            }
        }
    };
}

macro_rules! impl_approx_aligned {
    ($ty:ident, $count:literal, |$value:ident| $to_array:expr) => {
        impl AbsDiffEq for $ty {
            type Epsilon = f32;
            #[inline]
            fn default_epsilon() -> f32 {
                f32::default_epsilon()
            }
            #[inline]
            fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
                let $value = self;
                let a: [f32; $count] = $to_array;
                let $value = other;
                let b: [f32; $count] = $to_array;
                a.iter().zip(&b).all(|(a, b)| a.abs_diff_eq(b, epsilon))
            }
        }
        impl RelativeEq for $ty {
            #[inline]
            fn default_max_relative() -> f32 {
                f32::default_max_relative()
            }
            #[inline]
            fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
                let $value = self;
                let a: [f32; $count] = $to_array;
                let $value = other;
                let b: [f32; $count] = $to_array;
                a.iter()
                    .zip(&b)
                    .all(|(a, b)| a.relative_eq(b, epsilon, max_relative))
            }
        }
        impl UlpsEq for $ty {
            #[inline]
            fn default_max_ulps() -> u32 {
                f32::default_max_ulps()
            }
            #[inline]
            fn ulps_eq(&self, other: &Self, epsilon: f32, max_ulps: u32) -> bool {
                let $value = self;
                let a: [f32; $count] = $to_array;
                let $value = other;
                let b: [f32; $count] = $to_array;
                a.iter()
                    .zip(&b)
                    .all(|(a, b)| a.ulps_eq(b, epsilon, max_ulps))
            }
        }
    };
}

impl_approx!(GVec2<T: Copy>, 2, |v| v.to_array());
impl_approx!(GVec3<T: Copy>, 3, |v| v.to_array());
impl_approx!(GVec4<T: Copy>, 4, |v| v.to_array());

impl_approx!(GMat2<T: Real>, 4, |m| m.to_cols_array());
impl_approx!(GMat3<T: Real>, 9, |m| m.to_cols_array());
impl_approx!(GMat4<T: Real>, 16, |m| m.to_cols_array());

impl_approx!(GRot2<T: Real>, 2, |r| r.to_array());
impl_approx!(GRot3<T: Real>, 4, |r| r.to_array());

impl_approx!(GAffine2<T: Real>, 6, |a| a.to_cols_array());
impl_approx!(GAffine3<T: Real>, 12, |a| a.to_cols_array());

impl_approx!(GIso2<T: Real>, 4, |i| {
    let [cos, sin] = i.rotation.to_array();
    let [x, y] = i.translation.to_array();
    [cos, sin, x, y]
});
impl_approx!(GIso3<T: Real>, 7, |i| {
    let [x, y, z, w] = i.rotation.to_array();
    let [tx, ty, tz] = i.translation.to_array();
    [x, y, z, w, tx, ty, tz]
});

impl_approx_aligned!(Vec3A, 3, |v| v.to_array());
impl_approx_aligned!(Vec4A, 4, |v| v.to_array());
impl_approx_aligned!(Mat2A, 4, |m| m.to_cols_array());
impl_approx_aligned!(Mat3A, 9, |m| m.to_cols_array());
impl_approx_aligned!(Mat4A, 16, |m| m.to_cols_array());
impl_approx_aligned!(Rot3A, 4, |r| r.to_array());
impl_approx_aligned!(Affine2A, 6, |a| a.to_cols_array());
impl_approx_aligned!(Affine3A, 12, |a| a.to_cols_array());
impl_approx_aligned!(Iso3A, 7, |i| {
    let [x, y, z, w] = i.rotation.to_array();
    let [tx, ty, tz] = i.translation.to_array();
    [x, y, z, w, tx, ty, tz]
});

#[cfg(test)]
mod tests {
    use approx::{abs_diff_eq, abs_diff_ne, relative_eq, relative_ne, ulps_eq, ulps_ne};

    use crate::{
        affine::{Affine2, Affine3},
        isometry::{Iso2, Iso3},
        matrix::{DMat3, Mat2, Mat3, Mat4},
        rotation::{Rot2, Rot3},
        vector::{DVec2, Vec2, Vec3, Vec4},
    };

    macro_rules! test_approx {
        ($name:ident, $ty:ident, $value:expr, $offset:expr, $scale:literal) => {
            #[test]
            fn $name() {
                let value = $value;
                let near = value + $offset * $scale;
                let far = value + $offset;

                assert!(abs_diff_eq!(value, near, epsilon = 1e-6));
                assert!(abs_diff_ne!(value, far, epsilon = 1e-6));

                assert!(relative_eq!(value, near, epsilon = 1e-6));
                assert!(relative_ne!(value, far, epsilon = 1e-6));

                assert!(ulps_eq!(value, near));
                assert!(ulps_ne!(value, far));
            }
        };
    }

    test_approx!(vec2, Vec2, Vec2::new(1.0, 2.0), Vec2::X, 1e-8);
    test_approx!(vec3, Vec3, Vec3::new(1.0, 2.0, 3.0), Vec3::Y, 1e-8);
    test_approx!(vec4, Vec4, Vec4::new(1.0, 2.0, 3.0, 4.0), Vec4::Z, 1e-8);
    test_approx!(dvec2, DVec2, DVec2::new(1.0, 2.0), DVec2::X, 1e-17);

    test_approx!(
        mat2,
        Mat2,
        Mat2::IDENTITY,
        Mat2::from_cols_array(&[1.0; 4]),
        1e-8
    );
    test_approx!(
        mat3,
        Mat3,
        Mat3::IDENTITY,
        Mat3::from_cols_array(&[1.0; 9]),
        1e-8
    );
    test_approx!(
        mat4,
        Mat4,
        Mat4::IDENTITY,
        Mat4::from_cols_array(&[1.0; 16]),
        1e-8
    );
    test_approx!(
        dmat3,
        DMat3,
        DMat3::IDENTITY,
        DMat3::from_cols_array(&[1.0; 9]),
        1e-17
    );

    #[test]
    fn rotations() {
        let rot2 = Rot2::from_radians(1.0);
        assert!(abs_diff_eq!(rot2, Rot2::from_radians(1.0 + 1e-8)));
        assert!(abs_diff_ne!(rot2, Rot2::from_radians(1.5)));

        let rot3 = Rot3::from_axis_angle(Vec3::Z, 1.0);
        assert!(abs_diff_eq!(rot3, Rot3::from_axis_angle(Vec3::Z, 1.0)));
        assert!(abs_diff_ne!(rot3, Rot3::from_axis_angle(Vec3::Z, 1.5)));
    }

    #[test]
    fn affines_and_isometries() {
        assert!(abs_diff_eq!(Affine2::IDENTITY, Affine2::IDENTITY));
        assert!(abs_diff_ne!(Affine2::IDENTITY, Affine2::ZERO));

        assert!(abs_diff_eq!(Affine3::IDENTITY, Affine3::IDENTITY));
        assert!(abs_diff_ne!(Affine3::IDENTITY, Affine3::ZERO));

        let iso2 = Iso2::from_rotation_translation(Rot2::IDENTITY, Vec2::new(1.0, 2.0));
        assert!(abs_diff_eq!(iso2, iso2));
        assert!(abs_diff_ne!(iso2, Iso2::IDENTITY));

        let iso3 = Iso3::from_rotation_translation(Rot3::IDENTITY, Vec3::new(1.0, 2.0, 3.0));
        assert!(abs_diff_eq!(iso3, iso3));
        assert!(abs_diff_ne!(iso3, Iso3::IDENTITY));
    }
}
