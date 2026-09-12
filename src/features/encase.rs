use encase::matrix::{AsMutMatrixParts, AsRefMatrixParts, FromMatrixParts, MatrixScalar};
#[cfg(feature = "simd")]
use encase::private::{CreateFrom, MatrixMetadata, ReadFrom, ShaderSize, ShaderType, WriteInto};
use gnum::num::Real;

use crate::{
    matrix::{GMat2, GMat3, GMat4},
    vector::{GVec2, GVec3, GVec4},
};
#[cfg(feature = "simd")]
use crate::{
    matrix::{Mat2A, Mat3A, Mat4A},
    vector::{Vec3A, Vec4A},
};

encase::impl_vector!(2, GVec2<T>; (T: Copy); using AsRef AsMut From);
encase::impl_vector!(3, GVec3<T>; (T: Copy); using AsRef AsMut From);
encase::impl_vector!(4, GVec4<T>; (T: Copy); using AsRef AsMut From);
#[cfg(feature = "simd")]
encase::impl_vector!(3, Vec3A, f32; using AsRef AsMut From);
#[cfg(feature = "simd")]
encase::impl_vector!(4, Vec4A, f32; using AsRef AsMut From);

macro_rules! impl_matrix_parts {
    ($($ty:ident, $dim:literal, $count:literal);* $(;)?) => {
        $(
            impl<T: Real + MatrixScalar> AsRefMatrixParts<T, $dim, $dim> for $ty<T> {
                #[inline]
                fn as_ref_parts(&self) -> &[[T; $dim]; $dim] {
                    let elements: &[T; $count] = self.as_ref();

                    // SAFETY: `[T; C * R]` and `[[T; R]; C]` have the same layout.
                    unsafe { &*(elements as *const [T; $count] as *const [[T; $dim]; $dim]) }
                }
            }

            impl<T: Real + MatrixScalar> AsMutMatrixParts<T, $dim, $dim> for $ty<T> {
                #[inline]
                fn as_mut_parts(&mut self) -> &mut [[T; $dim]; $dim] {
                    let elements: &mut [T; $count] = self.as_mut();

                    // SAFETY: `[T; C * R]` and `[[T; R]; C]` have the same layout.
                    unsafe { &mut *(elements as *mut [T; $count] as *mut [[T; $dim]; $dim]) }
                }
            }

            impl<T: Real + MatrixScalar> FromMatrixParts<T, $dim, $dim> for $ty<T> {
                #[inline]
                fn from_parts(parts: [[T; $dim]; $dim]) -> Self {
                    Self::from_cols_array_2d(&parts)
                }
            }
        )*
    };
}

impl_matrix_parts! {
    GMat2, 2, 4;
    GMat3, 3, 9;
    GMat4, 4, 16;
}

encase::impl_matrix!(2, 2, GMat2<T>; (T: Real));
encase::impl_matrix!(3, 3, GMat3<T>; (T: Real));
encase::impl_matrix!(4, 4, GMat4<T>; (T: Real));

#[cfg(feature = "simd")]
macro_rules! impl_aligned_matrix {
    ($aligned:ident, $generic:ident) => {
        impl ShaderType for $aligned {
            type ExtraMetadata = MatrixMetadata;
            const METADATA: encase::private::Metadata<MatrixMetadata> =
                <$generic<f32> as ShaderType>::METADATA;
        }

        impl ShaderSize for $aligned {}

        impl WriteInto for $aligned {
            #[inline]
            fn write_into<B: encase::private::BufferMut>(
                &self,
                writer: &mut encase::private::Writer<B>,
            ) {
                <$generic<f32> as WriteInto>::write_into(&(*self).into(), writer);
            }
        }

        impl ReadFrom for $aligned {
            #[inline]
            fn read_from<B: encase::private::BufferRef>(
                &mut self,
                reader: &mut encase::private::Reader<B>,
            ) {
                let mut value: $generic<f32> = (*self).into();
                value.read_from(reader);
                *self = value.into();
            }
        }

        impl CreateFrom for $aligned {
            #[inline]
            fn create_from<B: encase::private::BufferRef>(
                reader: &mut encase::private::Reader<B>,
            ) -> Self {
                <$generic<f32> as CreateFrom>::create_from(reader).into()
            }
        }
    };
}

#[cfg(feature = "simd")]
impl_aligned_matrix!(Mat2A, GMat2);
#[cfg(feature = "simd")]
impl_aligned_matrix!(Mat3A, GMat3);
#[cfg(feature = "simd")]
impl_aligned_matrix!(Mat4A, GMat4);

#[cfg(test)]
mod tests {
    use encase::{ShaderSize, ShaderType, UniformBuffer};

    use crate::{
        matrix::{Mat2, Mat3, Mat4},
        vector::{IVec2, UVec4, Vec2, Vec3, Vec4},
    };

    /// Checks the WGSL size and alignment of a type, and that it survives
    /// a write and read round trip through a uniform buffer.
    macro_rules! test_encase {
        ($name:ident, $ty:ident, $value:expr, $size:literal, $alignment:literal) => {
            #[test]
            fn $name() {
                assert_eq!(<$ty>::SHADER_SIZE.get(), $size);
                assert_eq!(<$ty as ShaderType>::METADATA.alignment().get(), $alignment);

                let value: $ty = $value;

                // A buffer large enough for every type tested here.
                let mut buffer = UniformBuffer::new([0_u8; 64]);
                buffer.write(&value).unwrap();

                let read: $ty = buffer.create().unwrap();
                assert_eq!(read, value);
            }
        };
    }

    test_encase!(vec2, Vec2, Vec2::new(1.0, 2.0), 8, 8);
    test_encase!(vec3, Vec3, Vec3::new(1.0, 2.0, 3.0), 12, 16);
    test_encase!(vec4, Vec4, Vec4::new(1.0, 2.0, 3.0, 4.0), 16, 16);
    test_encase!(ivec2, IVec2, IVec2::new(1, 2), 8, 8);
    test_encase!(uvec4, UVec4, UVec4::new(1, 2, 3, 4), 16, 16);

    test_encase!(
        mat2,
        Mat2,
        Mat2::from_cols_array(&[1.0, 2.0, 3.0, 4.0]),
        16,
        8
    );
    test_encase!(mat3, Mat3, Mat3::IDENTITY, 48, 16);
    test_encase!(mat4, Mat4, Mat4::IDENTITY, 64, 16);

    #[test]
    fn matrix_parts_are_column_major() {
        use encase::matrix::AsRefMatrixParts;

        let matrix = Mat3::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let parts: &[[f32; 3]; 3] = AsRefMatrixParts::as_ref_parts(&matrix);

        assert_eq!(parts[0], [1.0, 2.0, 3.0]);
        assert_eq!(parts[1], [4.0, 5.0, 6.0]);
        assert_eq!(parts[2], [7.0, 8.0, 9.0]);
    }
}
