/// A marker trait for element types that never cause padding bytes
/// in the types provided by `gela`.
///
/// # Safety
///
/// The size of `Self` must be a non-zero multiple of 4 bytes.
pub unsafe trait UnpaddedElement {}

macro_rules! impl_unpadded_element {
    ($($t:ty),* $(,)?) => {
        $(
            // SAFETY: The size of the type is a non-zero multiple of 4 bytes.
            unsafe impl UnpaddedElement for $t {}
        )*
    };
}

impl_unpadded_element!(f32, f64, i32, u32, i64, u64, i128, u128);

// `isize` and `usize` are only guaranteed to be at least 2 bytes large,
// so they are only supported on targets where they are at least 4 bytes.
#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl_unpadded_element!(isize, usize);

impl_unpadded_element!(
    gimd::f32x4,
    gimd::f32x8,
    gimd::f64x2,
    gimd::f64x4,
    gimd::i32x4,
    gimd::i32x8,
    gimd::u32x4,
    gimd::u32x8,
);

#[cfg(test)]
mod tests {
    use super::UnpaddedElement;

    fn assert_unpadded<T: UnpaddedElement>() {
        let size = size_of::<T>();
        assert!(size != 0 && size.is_multiple_of(4));
    }

    #[test]
    fn element_sizes_are_multiples_of_four() {
        assert_unpadded::<f32>();
        assert_unpadded::<f64>();
        assert_unpadded::<i32>();
        assert_unpadded::<u32>();
        assert_unpadded::<i64>();
        assert_unpadded::<u64>();
        assert_unpadded::<i128>();
        assert_unpadded::<u128>();
        assert_unpadded::<isize>();
        assert_unpadded::<usize>();
    }

    #[test]
    fn simd_element_sizes_are_multiples_of_four() {
        assert_unpadded::<gimd::f32x4>();
        assert_unpadded::<gimd::f32x8>();
        assert_unpadded::<gimd::f64x2>();
        assert_unpadded::<gimd::f64x4>();
        assert_unpadded::<gimd::i32x4>();
        assert_unpadded::<gimd::i32x8>();
        assert_unpadded::<gimd::u32x4>();
        assert_unpadded::<gimd::u32x8>();
    }
}
