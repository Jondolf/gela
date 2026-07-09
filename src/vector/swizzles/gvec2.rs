use crate::vector::{GVec2, GVec3, GVec4, Vec2Swizzles};

impl<T: Copy> Vec2Swizzles for GVec2<T> {
    type Vec3 = GVec3<T>;
    type Vec4 = GVec4<T>;

    #[inline]
    fn xx(self) -> Self {
        Self {
            x: self.x,
            y: self.x,
        }
    }

    #[inline]
    fn yx(self) -> Self {
        Self {
            x: self.y,
            y: self.x,
        }
    }

    #[inline]
    fn yy(self) -> Self {
        Self {
            x: self.y,
            y: self.y,
        }
    }

    #[inline]
    fn xxx(self) -> GVec3<T> {
        GVec3::new(self.x, self.x, self.x)
    }

    #[inline]
    fn xxy(self) -> GVec3<T> {
        GVec3::new(self.x, self.x, self.y)
    }

    #[inline]
    fn xyx(self) -> GVec3<T> {
        GVec3::new(self.x, self.y, self.x)
    }

    #[inline]
    fn xyy(self) -> GVec3<T> {
        GVec3::new(self.x, self.y, self.y)
    }

    #[inline]
    fn yxx(self) -> GVec3<T> {
        GVec3::new(self.y, self.x, self.x)
    }

    #[inline]
    fn yxy(self) -> GVec3<T> {
        GVec3::new(self.y, self.x, self.y)
    }

    #[inline]
    fn yyx(self) -> GVec3<T> {
        GVec3::new(self.y, self.y, self.x)
    }

    #[inline]
    fn yyy(self) -> GVec3<T> {
        GVec3::new(self.y, self.y, self.y)
    }

    #[inline]
    fn xxxx(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.x, self.x)
    }

    #[inline]
    fn xxxy(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.x, self.y)
    }

    #[inline]
    fn xxyx(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.y, self.x)
    }

    #[inline]
    fn xxyy(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.y, self.y)
    }

    #[inline]
    fn xyxx(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.x, self.x)
    }

    #[inline]
    fn xyxy(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.x, self.y)
    }

    #[inline]
    fn xyyx(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.y, self.x)
    }

    #[inline]
    fn xyyy(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.y, self.y)
    }

    #[inline]
    fn yxxx(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.x, self.x)
    }

    #[inline]
    fn yxxy(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.x, self.y)
    }

    #[inline]
    fn yxyx(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.y, self.x)
    }

    #[inline]
    fn yxyy(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.y, self.y)
    }

    #[inline]
    fn yyxx(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.x, self.x)
    }

    #[inline]
    fn yyxy(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.x, self.y)
    }

    #[inline]
    fn yyyx(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.y, self.x)
    }

    #[inline]
    fn yyyy(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.y, self.y)
    }
}
