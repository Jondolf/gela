use gnum::num::Real;
use rand::{
    Rng, RngExt,
    distr::{
        Distribution, StandardUniform,
        uniform::{Error, SampleBorrow, SampleUniform, UniformSampler},
    },
};

use crate::{
    affine::{Affine2A, Affine3A, GAffine2, GAffine3},
    isometry::{GIso2, GIso3, Iso3A},
    matrix::{GMat2, GMat3, GMat4, Mat2A, Mat3A, Mat4A},
    rotation::{GRot2, GRot3, Rot3A},
    vector::{BVec3A, BVec4A, GVec2, GVec3, GVec4, Vec3A, Vec4A},
};

macro_rules! impl_standard_uniform {
    (
        $ty:ident<$elem:ident: $bound:ident>,
        $count:literal,
        |$array:ident| $from_array:expr $(,)?
    ) => {
        impl<$elem: $bound> Distribution<$ty<$elem>> for StandardUniform
        where
            StandardUniform: Distribution<$elem>,
        {
            #[inline]
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $ty<$elem> {
                let $array: [$elem; $count] = core::array::from_fn(|_| rng.random());
                $from_array
            }
        }
    };
}

macro_rules! impl_standard_uniform_aligned {
    ($ty:ident, $elem:ty, $count:literal, |$array:ident| $from_array:expr) => {
        impl Distribution<$ty> for StandardUniform
        where
            StandardUniform: Distribution<$elem>,
        {
            #[inline]
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $ty {
                let $array: [$elem; $count] = core::array::from_fn(|_| rng.random());
                $from_array
            }
        }
    };
}

impl_standard_uniform!(GVec2<T: Copy>, 2, |a| GVec2::from_array(a));
impl_standard_uniform!(GVec3<T: Copy>, 3, |a| GVec3::from_array(a));
impl_standard_uniform!(GVec4<T: Copy>, 4, |a| GVec4::from_array(a));

impl_standard_uniform!(GMat2<T: Real>, 4, |a| GMat2::from_cols_array(&a));
impl_standard_uniform!(GMat3<T: Real>, 9, |a| GMat3::from_cols_array(&a));
impl_standard_uniform!(GMat4<T: Real>, 16, |a| GMat4::from_cols_array(&a));

impl_standard_uniform!(GAffine2<T: Real>, 6, |a| GAffine2::from_cols_array(&a));
impl_standard_uniform!(GAffine3<T: Real>, 12, |a| GAffine3::from_cols_array(&a));

impl<T: Real + SampleUniform + PartialOrd> Distribution<GRot2<T>> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> GRot2<T> {
        GRot2::from_radians(rng.random_range(T::ZERO..T::TAU))
    }
}

impl<T: Real + SampleUniform + PartialOrd> Distribution<GRot3<T>> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> GRot3<T> {
        let z = rng.random_range(T::NEG_ONE..=T::ONE);
        let (sin, cos) = rng.random_range(T::ZERO..T::TAU).sin_cos();
        let radius = (T::ONE - z * z).sqrt();
        let axis = GVec3::new(radius * cos, radius * sin, z);

        GRot3::from_axis_angle(axis, rng.random_range(T::ZERO..T::TAU))
    }
}

impl<T: Real + SampleUniform + PartialOrd> Distribution<GIso2<T>> for StandardUniform
where
    StandardUniform: Distribution<T>,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> GIso2<T> {
        GIso2::from_rotation_translation(rng.random::<GRot2<T>>(), rng.random::<GVec2<T>>())
    }
}

impl<T: Real + SampleUniform + PartialOrd> Distribution<GIso3<T>> for StandardUniform
where
    StandardUniform: Distribution<T>,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> GIso3<T> {
        GIso3::from_rotation_translation(rng.random::<GRot3<T>>(), rng.random::<GVec3<T>>())
    }
}

impl_standard_uniform_aligned!(BVec3A, bool, 3, |a| BVec3A::from_array(a));
impl_standard_uniform_aligned!(BVec4A, bool, 4, |a| BVec4A::from_array(a));
impl_standard_uniform_aligned!(Vec3A, f32, 3, |a| Vec3A::from_array(a));
impl_standard_uniform_aligned!(Vec4A, f32, 4, |a| Vec4A::from_array(a));
impl_standard_uniform_aligned!(Mat2A, f32, 4, |a| Mat2A::from_cols_array(&a));
impl_standard_uniform_aligned!(Mat3A, f32, 9, |a| Mat3A::from_cols_array(&a));
impl_standard_uniform_aligned!(Mat4A, f32, 16, |a| Mat4A::from_cols_array(&a));
impl_standard_uniform_aligned!(Affine2A, f32, 6, |a| Affine2A::from_cols_array(&a));
impl_standard_uniform_aligned!(Affine3A, f32, 12, |a| Affine3A::from_cols_array(&a));

impl Distribution<Rot3A> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Rot3A {
        rng.random::<GRot3<f32>>().into()
    }
}

impl Distribution<Iso3A> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Iso3A {
        rng.random::<GIso3<f32>>().into()
    }
}

macro_rules! impl_sample_uniform {
    (
        $ty:ident<$elem:ident>,
        $sampler:ident,
        [$($field:ident),* $(,)?] $(,)?
    ) => {
        #[doc = concat!("A [`UniformSampler`] for [`", stringify!($ty), "`] values.")]
        ///
        /// Each element is sampled from its own range.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $sampler<S> {
            $($field: S,)*
        }

        impl<$elem: Copy + SampleUniform> SampleUniform for $ty<$elem> {
            type Sampler = $sampler<$elem::Sampler>;
        }

        impl<S: UniformSampler> UniformSampler for $sampler<S>
        where
            S::X: Copy + SampleUniform,
        {
            type X = $ty<S::X>;

            fn new<B1, B2>(low: B1, high: B2) -> Result<Self, Error>
            where
                B1: SampleBorrow<Self::X> + Sized,
                B2: SampleBorrow<Self::X> + Sized,
            {
                let low = *low.borrow();
                let high = *high.borrow();

                Ok(Self {
                    $($field: S::new(low.$field, high.$field)?,)*
                })
            }

            fn new_inclusive<B1, B2>(low: B1, high: B2) -> Result<Self, Error>
            where
                B1: SampleBorrow<Self::X> + Sized,
                B2: SampleBorrow<Self::X> + Sized,
            {
                let low = *low.borrow();
                let high = *high.borrow();

                Ok(Self {
                    $($field: S::new_inclusive(low.$field, high.$field)?,)*
                })
            }

            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
                $ty::new($(self.$field.sample(rng)),*)
            }

            fn sample_single<R: Rng + ?Sized, B1, B2>(
                low: B1,
                high: B2,
                rng: &mut R,
            ) -> Result<Self::X, Error>
            where
                B1: SampleBorrow<Self::X> + Sized,
                B2: SampleBorrow<Self::X> + Sized,
            {
                let low = *low.borrow();
                let high = *high.borrow();

                Ok($ty::new(
                    $(S::sample_single(low.$field, high.$field, rng)?),*
                ))
            }

            fn sample_single_inclusive<R: Rng + ?Sized, B1, B2>(
                low: B1,
                high: B2,
                rng: &mut R,
            ) -> Result<Self::X, Error>
            where
                B1: SampleBorrow<Self::X> + Sized,
                B2: SampleBorrow<Self::X> + Sized,
            {
                let low = *low.borrow();
                let high = *high.borrow();

                Ok($ty::new(
                    $(S::sample_single_inclusive(low.$field, high.$field, rng)?),*
                ))
            }
        }
    };
}

impl_sample_uniform!(GVec2<T>, UniformGVec2, [x, y]);
impl_sample_uniform!(GVec3<T>, UniformGVec3, [x, y, z]);
impl_sample_uniform!(GVec4<T>, UniformGVec4, [x, y, z, w]);

macro_rules! impl_sample_uniform_aligned {
    ($ty:ident, $sampler:ident, [$($field:ident),+ $(,)?]) => {
        #[doc = concat!("A [`UniformSampler`] for [`", stringify!($ty), "`] values.")]
        ///
        /// Each element is sampled from its own range.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $sampler<S> { $($field: S,)+ }

        impl SampleUniform for $ty {
            type Sampler = $sampler<<f32 as SampleUniform>::Sampler>;
        }

        impl<S: UniformSampler<X = f32>> UniformSampler for $sampler<S> {
            type X = $ty;

            fn new<B1: SampleBorrow<$ty>, B2: SampleBorrow<$ty>>(low: B1, high: B2) -> Result<Self, Error> {
                let low = *low.borrow(); let high = *high.borrow();
                Ok(Self { $($field: S::new(low.$field, high.$field)?,)+ })
            }

            fn new_inclusive<B1: SampleBorrow<$ty>, B2: SampleBorrow<$ty>>(low: B1, high: B2) -> Result<Self, Error> {
                let low = *low.borrow(); let high = *high.borrow();
                Ok(Self { $($field: S::new_inclusive(low.$field, high.$field)?,)+ })
            }

            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $ty {
                $ty::new($(self.$field.sample(rng)),+)
            }

            fn sample_single<R: Rng + ?Sized, B1: SampleBorrow<$ty>, B2: SampleBorrow<$ty>>(
                low: B1, high: B2, rng: &mut R,
            ) -> Result<$ty, Error> {
                let low = *low.borrow(); let high = *high.borrow();
                Ok($ty::new($(S::sample_single(low.$field, high.$field, rng)?),+))
            }

            fn sample_single_inclusive<R: Rng + ?Sized, B1: SampleBorrow<$ty>, B2: SampleBorrow<$ty>>(
                low: B1, high: B2, rng: &mut R,
            ) -> Result<$ty, Error> {
                let low = *low.borrow(); let high = *high.borrow();
                Ok($ty::new($(S::sample_single_inclusive(low.$field, high.$field, rng)?),+))
            }
        }
    };
}

impl_sample_uniform_aligned!(Vec3A, UniformVec3A, [x, y, z]);
impl_sample_uniform_aligned!(Vec4A, UniformVec4A, [x, y, z, w]);

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use crate::{
        affine::{Affine2, Affine3},
        isometry::{Iso2, Iso3},
        matrix::{DMat3, Mat2, Mat3, Mat4},
        rotation::{DRot3, Rot2, Rot3},
        vector::{DVec2, IVec3, UVec2, Vec2, Vec3, Vec4},
    };

    fn seeded_rng() -> SmallRng {
        SmallRng::seed_from_u64(0)
    }

    /// Checks that a type can be sampled, and that sampling is deterministic for a seed.
    ///
    /// Types that are sampled with trigonometric operations must pass `non_deterministic_math`,
    /// as Miri deliberately makes those operations return slightly different results on every call.
    macro_rules! test_standard_uniform {
        ($name:ident, $ty:ident, non_deterministic_math) => {
            test_standard_uniform!(
                $name,
                $ty,
                #[cfg_attr(miri, ignore = "Miri makes trigonometric operations non-deterministic")]
            );
        };
        ($name:ident, $ty:ident) => {
            test_standard_uniform!($name, $ty,);
        };
        ($name:ident, $ty:ident, $(#[$attr:meta])*) => {
            #[test]
            $(#[$attr])*
            fn $name() {
                let a: $ty = seeded_rng().random();
                let b: $ty = seeded_rng().random();
                assert_eq!(a, b);
            }
        };
    }

    test_standard_uniform!(vec2, Vec2);
    test_standard_uniform!(vec3, Vec3);
    test_standard_uniform!(vec4, Vec4);
    test_standard_uniform!(dvec2, DVec2);
    test_standard_uniform!(ivec3, IVec3);
    test_standard_uniform!(uvec2, UVec2);

    test_standard_uniform!(mat2, Mat2);
    test_standard_uniform!(mat3, Mat3);
    test_standard_uniform!(mat4, Mat4);
    test_standard_uniform!(dmat3, DMat3);

    test_standard_uniform!(rot2, Rot2, non_deterministic_math);
    test_standard_uniform!(rot3, Rot3, non_deterministic_math);
    test_standard_uniform!(drot3, DRot3, non_deterministic_math);

    test_standard_uniform!(affine2, Affine2);
    test_standard_uniform!(affine3, Affine3);

    test_standard_uniform!(iso2, Iso2, non_deterministic_math);
    test_standard_uniform!(iso3, Iso3, non_deterministic_math);

    #[test]
    fn vectors_are_sampled_element_wise() {
        let vector: Vec3 = seeded_rng().random();
        let elements = seeded_rng().random::<[f32; 3]>();

        assert_eq!(vector, Vec3::from_array(elements));
    }

    #[test]
    fn rotations_are_normalized() {
        let mut rng = seeded_rng();

        for _ in 0..100 {
            let rot2: Rot2 = rng.random();
            assert!(rot2.is_normalized(1e-5));

            let rot3: Rot3 = rng.random();
            assert!(rot3.is_normalized(1e-5));
        }
    }

    #[test]
    fn ranges() {
        use rand::distr::{Distribution, Uniform};

        let mut rng = seeded_rng();

        let low = Vec3::new(-1.0, 0.0, 5.0);
        let high = Vec3::new(1.0, 10.0, 6.0);

        let exclusive = Uniform::new(low, high).unwrap();
        let inclusive = Uniform::new_inclusive(low, high).unwrap();

        for _ in 0..100 {
            let vector = exclusive.sample(&mut rng);
            assert!(vector.cmpge(low).all() && vector.cmplt(high).all());

            let vector = inclusive.sample(&mut rng);
            assert!(vector.cmpge(low).all() && vector.cmple(high).all());
        }

        let integers = Uniform::new(IVec3::new(-5, 0, 1), IVec3::new(5, 1, 2))
            .unwrap()
            .sample(&mut rng);
        assert_eq!(integers.y, 0);
        assert_eq!(integers.z, 1);
    }

    #[test]
    fn single_ranges() {
        use rand::distr::uniform::{SampleUniform, UniformSampler};

        type Sampler = <Vec3 as SampleUniform>::Sampler;

        let mut rng = seeded_rng();

        let low = Vec3::new(-1.0, 0.0, 5.0);
        let high = Vec3::new(1.0, 10.0, 6.0);

        for _ in 0..100 {
            let vector = Sampler::sample_single(low, high, &mut rng).unwrap();
            assert!(vector.cmpge(low).all() && vector.cmplt(high).all());

            let vector = Sampler::sample_single_inclusive(low, high, &mut rng).unwrap();
            assert!(vector.cmpge(low).all() && vector.cmple(high).all());
        }

        assert!(Sampler::sample_single(high, low, &mut rng).is_err());
    }

    #[test]
    fn empty_range_is_an_error() {
        use rand::distr::Uniform;

        assert!(Uniform::new(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)).is_err());
        assert!(Uniform::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)).is_ok());
    }
}
