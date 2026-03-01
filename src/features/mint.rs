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

impl<T: Copy> IntoMint for GVec3<T> {
    type MintType = mint::Vector3<T>;
}

impl<T: Copy> From<Point3<T>> for GVec3<T> {
    #[inline]
    fn from(p: Point3<T>) -> Self {
        Self {
            x: p.x,
            y: p.y,
            z: p.z,
        }
    }
}

impl<T: Copy> From<GVec3<T>> for Point3<T> {
    #[inline]
    fn from(v: GVec3<T>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl<T: Copy> From<Vector3<T>> for GVec3<T> {
    #[inline]
    fn from(v: Vector3<T>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl<T: Copy> From<GVec3<T>> for Vector3<T> {
    #[inline]
    fn from(v: GVec3<T>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl<T: Copy> IntoMint for GVec4<T> {
    type MintType = mint::Vector4<T>;
}

impl<T: Copy> From<Point4<T>> for GVec4<T> {
    #[inline]
    fn from(p: Point4<T>) -> Self {
        Self {
            x: p.x,
            y: p.y,
            z: p.z,
            w: p.w,
        }
    }
}

impl<T: Copy> From<GVec4<T>> for Point4<T> {
    #[inline]
    fn from(v: GVec4<T>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: v.w,
        }
    }
}

impl<T: Copy> From<Vector4<T>> for GVec4<T> {
    #[inline]
    fn from(v: Vector4<T>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: v.w,
        }
    }
}

impl<T: Copy> From<GVec4<T>> for Vector4<T> {
    #[inline]
    fn from(v: GVec4<T>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: v.w,
        }
    }
}
