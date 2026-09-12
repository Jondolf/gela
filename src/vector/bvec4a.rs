use crate::vector::GVec4;

use core::{fmt, hash::Hash, ops::*};

use gimd::i32x4;
use gnum::simd::{MaskCast, MaskLike, SimdLike};

/// Creates a 4-dimensional boolean vector mask, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for mask operations,
/// which can provide better performance than [`BVec4`](crate::vector::BVec4).
#[inline(always)]
#[must_use]
pub const fn bvec4a(x: bool, y: bool, z: bool, w: bool) -> BVec4A {
    BVec4A::new(x, y, z, w)
}

/// A 4-dimensional boolean vector mask, using SIMD vector types
/// on supported platforms.
///
/// The type is 16-byte aligned and uses SIMD instructions for mask operations,
/// which can provide better performance than [`BVec4`](crate::vector::BVec4).
#[derive(Clone, Copy)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[cfg_attr(
    feature = "zerocopy",
    derive(
        zerocopy_derive::FromBytes,
        zerocopy_derive::Immutable,
        zerocopy_derive::IntoBytes,
        zerocopy_derive::KnownLayout
    )
)]
#[repr(transparent)]
pub struct BVec4A(i32x4);

const MASK: [i32; 2] = [0, -1];

/// # Boolean Constants
impl BVec4A {
    /// All `true`.
    pub const TRUE: Self = Self::splat(true);

    /// All `false`.
    pub const FALSE: Self = Self::splat(false);
}

/// # Construction
impl BVec4A {
    /// Creates a new vector mask.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: bool, y: bool, z: bool, w: bool) -> Self {
        Self(i32x4::from_array([
            MASK[x as usize],
            MASK[y as usize],
            MASK[z as usize],
            MASK[w as usize],
        ]))
    }

    /// Creates a new vector mask with all elements set to `v`.
    #[inline(always)]
    #[must_use]
    pub const fn splat(v: bool) -> Self {
        Self::new(v, v, v, v)
    }

    /// Creates a new vector mask from an array.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(arr: [bool; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }

    /// Returns the vector mask as an array.
    #[inline]
    #[must_use]
    pub fn to_array(self) -> [bool; 4] {
        let bits = self.bitmask();
        [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0]
    }

    /// Returns a bitmask with the lowest 4 bits set from the elements of `self`.
    ///
    /// A true element results in a `1` bit and a false element in a `0` bit. Element `x` goes
    /// into the lowest bit, element `y` into the second-lowest bit, and so on.
    #[inline(always)]
    #[must_use]
    pub fn bitmask(self) -> u32 {
        MaskLike::to_bitmask(self.to_mask()) as u32
    }

    /// Returns `true` if all elements of `self` are true, and `false` otherwise.
    #[inline(always)]
    #[must_use]
    pub fn all(self) -> bool {
        self.bitmask() == 0xf
    }

    /// Returns `true` if any element of `self` is true, and `false` otherwise.
    #[inline(always)]
    #[must_use]
    pub fn any(self) -> bool {
        self.bitmask() != 0
    }

    /// Tests the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    #[must_use]
    pub fn test(self, index: usize) -> bool {
        assert!(index < 4, "index out of bounds");
        MaskLike::test(&self.to_mask(), index)
    }

    /// Sets the element at `index` to `value`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        assert!(index < 4, "index out of bounds");
        SimdLike::replace(&mut self.0, index, MASK[value as usize]);
    }

    #[inline(always)]
    pub(crate) fn from_mask(mask: <gimd::f32x4 as SimdLike>::Bool) -> Self {
        Self(i32x4::from_inner(MaskCast::to_int(mask)))
    }

    #[inline(always)]
    pub(crate) fn to_mask(self) -> <gimd::f32x4 as SimdLike>::Bool {
        MaskCast::from_int(self.0.into_inner())
    }
}

impl Default for BVec4A {
    #[inline(always)]
    fn default() -> Self {
        Self::FALSE
    }
}

impl PartialEq for BVec4A {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.bitmask() == rhs.bitmask()
    }
}

impl PartialEq<GVec4<bool>> for BVec4A {
    #[inline]
    fn eq(&self, rhs: &GVec4<bool>) -> bool {
        self.to_array() == rhs.to_array()
    }
}

impl PartialEq<BVec4A> for GVec4<bool> {
    #[inline]
    fn eq(&self, rhs: &BVec4A) -> bool {
        self.to_array() == rhs.to_array()
    }
}

impl Eq for BVec4A {}

impl Hash for BVec4A {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.bitmask().hash(state);
    }
}

macro_rules! impl_binary_op {
    ($($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident);* $(;)?) => {
        $(
            impl $trait for BVec4A {
                type Output = Self;

                #[inline(always)]
                fn $method(self, rhs: Self) -> Self {
                    Self($trait::$method(self.0, rhs.0))
                }
            }

            impl $trait<&BVec4A> for BVec4A {
                type Output = Self;

                #[inline(always)]
                fn $method(self, rhs: &Self) -> Self {
                    self.$method(*rhs)
                }
            }

            impl $trait<BVec4A> for &BVec4A {
                type Output = BVec4A;

                #[inline(always)]
                fn $method(self, rhs: BVec4A) -> BVec4A {
                    (*self).$method(rhs)
                }
            }

            impl $trait<&BVec4A> for &BVec4A {
                type Output = BVec4A;

                #[inline(always)]
                fn $method(self, rhs: &BVec4A) -> BVec4A {
                    (*self).$method(*rhs)
                }
            }

            impl $assign_trait for BVec4A {
                #[inline(always)]
                fn $assign_method(&mut self, rhs: Self) {
                    *self = self.$method(rhs);
                }
            }

            impl $assign_trait<&BVec4A> for BVec4A {
                #[inline(always)]
                fn $assign_method(&mut self, rhs: &Self) {
                    self.$assign_method(*rhs);
                }
            }
        )*
    };
}

impl_binary_op!(
    BitAnd, bitand, BitAndAssign, bitand_assign;
    BitOr, bitor, BitOrAssign, bitor_assign;
    BitXor, bitxor, BitXorAssign, bitxor_assign;
);

impl Not for BVec4A {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl Not for &BVec4A {
    type Output = BVec4A;

    #[inline(always)]
    fn not(self) -> BVec4A {
        !*self
    }
}

impl From<GVec4<bool>> for BVec4A {
    #[inline(always)]
    fn from(v: GVec4<bool>) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }
}

impl From<BVec4A> for GVec4<bool> {
    #[inline]
    fn from(v: BVec4A) -> Self {
        Self::from_array(v.to_array())
    }
}

impl From<[bool; 4]> for BVec4A {
    #[inline(always)]
    fn from(arr: [bool; 4]) -> Self {
        Self::from_array(arr)
    }
}

impl From<BVec4A> for [bool; 4] {
    #[inline]
    fn from(v: BVec4A) -> Self {
        v.to_array()
    }
}

impl From<BVec4A> for [u32; 4] {
    #[inline]
    fn from(v: BVec4A) -> Self {
        let [x, y, z, w] = v.to_array();
        [
            MASK[x as usize] as u32,
            MASK[y as usize] as u32,
            MASK[z as usize] as u32,
            MASK[w as usize] as u32,
        ]
    }
}

impl fmt::Display for BVec4A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [x, y, z, w] = self.to_array();
        write!(f, "[{x}, {y}, {z}, {w}]")
    }
}

impl fmt::Debug for BVec4A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arr: [u32; 4] = (*self).into();
        f.debug_tuple("BVec4A")
            .field(&format_args!("{:#x}", arr[0]))
            .field(&format_args!("{:#x}", arr[1]))
            .field(&format_args!("{:#x}", arr[2]))
            .field(&format_args!("{:#x}", arr[3]))
            .finish()
    }
}
