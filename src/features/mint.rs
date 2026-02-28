use crate::*;
use mint::*;

impl<T: Copy> IntoMint for GVec2<T> {
    type MintType = mint::Vector2<T>;
}

impl<T: Copy> From<Point2<T>> for GVec2<T> {
    #[inline]
    fn from(p: Point2<T>) -> Self {
        Self { x: p.x, y: p.y }
    }
}

impl<T: Copy> From<GVec2<T>> for Point2<T> {
    #[inline]
    fn from(v: GVec2<T>) -> Self {
        Self { x: v.x, y: v.y }
    }
}

impl<T: Copy> From<Vector2<T>> for GVec2<T> {
    #[inline]
    fn from(v: Vector2<T>) -> Self {
        Self { x: v.x, y: v.y }
    }
}

impl<T: Copy> From<GVec2<T>> for Vector2<T> {
    #[inline]
    fn from(v: GVec2<T>) -> Self {
        Self { x: v.x, y: v.y }
    }
}
