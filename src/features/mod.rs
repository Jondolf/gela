//! Implementations of traits from optional dependencies.

#[cfg(feature = "approx")]
mod approx;
#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "bytemuck")]
mod bytemuck;
#[cfg(feature = "encase")]
mod encase;
#[cfg(feature = "mint")]
mod mint;
#[cfg(feature = "rand")]
mod rand;
#[cfg(feature = "rkyv")]
mod rkyv;
#[cfg(feature = "serde")]
mod serde;
#[cfg(feature = "speedy")]
mod speedy;
#[cfg(feature = "zerocopy")]
mod zerocopy;

#[cfg(any(feature = "bytemuck", feature = "zerocopy"))]
mod unpadded;

#[cfg(feature = "rand")]
pub use rand::{UniformGVec2, UniformGVec3, UniformGVec4, UniformVec3A, UniformVec4A};
#[cfg(any(feature = "bytemuck", feature = "zerocopy"))]
pub use unpadded::UnpaddedElement;
