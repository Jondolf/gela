use crate::vector::{GVec2, GVec3, GVec4, Vec3Swizzles};

impl<T: Copy> Vec3Swizzles for GVec3<T> {
    type Vec2 = GVec2<T>;
    type Vec4 = GVec4<T>;

    #[inline]
    fn xx(self) -> GVec2<T> {
        GVec2::new(self.x, self.x)
    }

    #[inline]
    fn xy(self) -> GVec2<T> {
        GVec2::new(self.x, self.y)
    }

    #[inline]
    fn with_xy(self, rhs: GVec2<T>) -> Self {
        Self::new(rhs.x, rhs.y, self.z)
    }

    #[inline]
    fn xz(self) -> GVec2<T> {
        GVec2::new(self.x, self.z)
    }

    #[inline]
    fn with_xz(self, rhs: GVec2<T>) -> Self {
        Self::new(rhs.x, self.y, rhs.y)
    }

    #[inline]
    fn yx(self) -> GVec2<T> {
        GVec2::new(self.y, self.x)
    }

    #[inline]
    fn with_yx(self, rhs: GVec2<T>) -> Self {
        Self::new(rhs.y, rhs.x, self.z)
    }

    #[inline]
    fn yy(self) -> GVec2<T> {
        GVec2::new(self.y, self.y)
    }

    #[inline]
    fn yz(self) -> GVec2<T> {
        GVec2::new(self.y, self.z)
    }

    #[inline]
    fn with_yz(self, rhs: GVec2<T>) -> Self {
        Self::new(self.x, rhs.x, rhs.y)
    }

    #[inline]
    fn zx(self) -> GVec2<T> {
        GVec2::new(self.z, self.x)
    }

    #[inline]
    fn with_zx(self, rhs: GVec2<T>) -> Self {
        Self::new(rhs.y, self.y, rhs.x)
    }

    #[inline]
    fn zy(self) -> GVec2<T> {
        GVec2::new(self.z, self.y)
    }

    #[inline]
    fn with_zy(self, rhs: GVec2<T>) -> Self {
        Self::new(self.x, rhs.y, rhs.x)
    }

    #[inline]
    fn zz(self) -> GVec2<T> {
        GVec2::new(self.z, self.z)
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
    fn xxz(self) -> GVec3<T> {
        GVec3::new(self.x, self.x, self.z)
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
    fn xzx(self) -> GVec3<T> {
        GVec3::new(self.x, self.z, self.x)
    }

    #[inline]
    fn xzy(self) -> GVec3<T> {
        GVec3::new(self.x, self.z, self.y)
    }

    #[inline]
    fn xzz(self) -> GVec3<T> {
        GVec3::new(self.x, self.z, self.z)
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
    fn yxz(self) -> GVec3<T> {
        GVec3::new(self.y, self.x, self.z)
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
    fn yyz(self) -> GVec3<T> {
        GVec3::new(self.y, self.y, self.z)
    }

    #[inline]
    fn yzx(self) -> GVec3<T> {
        GVec3::new(self.y, self.z, self.x)
    }

    #[inline]
    fn yzy(self) -> GVec3<T> {
        GVec3::new(self.y, self.z, self.y)
    }

    #[inline]
    fn yzz(self) -> GVec3<T> {
        GVec3::new(self.y, self.z, self.z)
    }

    #[inline]
    fn zxx(self) -> GVec3<T> {
        GVec3::new(self.z, self.x, self.x)
    }

    #[inline]
    fn zxy(self) -> GVec3<T> {
        GVec3::new(self.z, self.x, self.y)
    }

    #[inline]
    fn zxz(self) -> GVec3<T> {
        GVec3::new(self.z, self.x, self.z)
    }

    #[inline]
    fn zyx(self) -> GVec3<T> {
        GVec3::new(self.z, self.y, self.x)
    }

    #[inline]
    fn zyy(self) -> GVec3<T> {
        GVec3::new(self.z, self.y, self.y)
    }

    #[inline]
    fn zyz(self) -> GVec3<T> {
        GVec3::new(self.z, self.y, self.z)
    }

    #[inline]
    fn zzx(self) -> GVec3<T> {
        GVec3::new(self.z, self.z, self.x)
    }

    #[inline]
    fn zzy(self) -> GVec3<T> {
        GVec3::new(self.z, self.z, self.y)
    }

    #[inline]
    fn zzz(self) -> GVec3<T> {
        GVec3::new(self.z, self.z, self.z)
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
    fn xxxz(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.x, self.z)
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
    fn xxyz(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.y, self.z)
    }

    #[inline]
    fn xxzx(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.z, self.x)
    }

    #[inline]
    fn xxzy(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.z, self.y)
    }

    #[inline]
    fn xxzz(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.z, self.z)
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
    fn xyxz(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.x, self.z)
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
    fn xyyz(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.y, self.z)
    }

    #[inline]
    fn xyzx(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.z, self.x)
    }

    #[inline]
    fn xyzy(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.z, self.y)
    }

    #[inline]
    fn xyzz(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.z, self.z)
    }

    #[inline]
    fn xzxx(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.x, self.x)
    }

    #[inline]
    fn xzxy(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.x, self.y)
    }

    #[inline]
    fn xzxz(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.x, self.z)
    }

    #[inline]
    fn xzyx(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.y, self.x)
    }

    #[inline]
    fn xzyy(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.y, self.y)
    }

    #[inline]
    fn xzyz(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.y, self.z)
    }

    #[inline]
    fn xzzx(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.z, self.x)
    }

    #[inline]
    fn xzzy(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.z, self.y)
    }

    #[inline]
    fn xzzz(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.z, self.z)
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
    fn yxxz(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.x, self.z)
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
    fn yxyz(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.y, self.z)
    }

    #[inline]
    fn yxzx(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.z, self.x)
    }

    #[inline]
    fn yxzy(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.z, self.y)
    }

    #[inline]
    fn yxzz(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.z, self.z)
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
    fn yyxz(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.x, self.z)
    }

    #[inline]
    fn yyyx(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.y, self.x)
    }

    #[inline]
    fn yyyy(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.y, self.y)
    }

    #[inline]
    fn yyyz(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.y, self.z)
    }

    #[inline]
    fn yyzx(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.z, self.x)
    }

    #[inline]
    fn yyzy(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.z, self.y)
    }

    #[inline]
    fn yyzz(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.z, self.z)
    }

    #[inline]
    fn yzxx(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.x, self.x)
    }

    #[inline]
    fn yzxy(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.x, self.y)
    }

    #[inline]
    fn yzxz(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.x, self.z)
    }

    #[inline]
    fn yzyx(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.y, self.x)
    }

    #[inline]
    fn yzyy(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.y, self.y)
    }

    #[inline]
    fn yzyz(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.y, self.z)
    }

    #[inline]
    fn yzzx(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.z, self.x)
    }

    #[inline]
    fn yzzy(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.z, self.y)
    }

    #[inline]
    fn yzzz(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.z, self.z)
    }

    #[inline]
    fn zxxx(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.x, self.x)
    }

    #[inline]
    fn zxxy(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.x, self.y)
    }

    #[inline]
    fn zxxz(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.x, self.z)
    }

    #[inline]
    fn zxyx(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.y, self.x)
    }

    #[inline]
    fn zxyy(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.y, self.y)
    }

    #[inline]
    fn zxyz(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.y, self.z)
    }

    #[inline]
    fn zxzx(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.z, self.x)
    }

    #[inline]
    fn zxzy(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.z, self.y)
    }

    #[inline]
    fn zxzz(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.z, self.z)
    }

    #[inline]
    fn zyxx(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.x, self.x)
    }

    #[inline]
    fn zyxy(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.x, self.y)
    }

    #[inline]
    fn zyxz(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.x, self.z)
    }

    #[inline]
    fn zyyx(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.y, self.x)
    }

    #[inline]
    fn zyyy(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.y, self.y)
    }

    #[inline]
    fn zyyz(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.y, self.z)
    }

    #[inline]
    fn zyzx(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.z, self.x)
    }

    #[inline]
    fn zyzy(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.z, self.y)
    }

    #[inline]
    fn zyzz(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.z, self.z)
    }

    #[inline]
    fn zzxx(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.x, self.x)
    }

    #[inline]
    fn zzxy(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.x, self.y)
    }

    #[inline]
    fn zzxz(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.x, self.z)
    }

    #[inline]
    fn zzyx(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.y, self.x)
    }

    #[inline]
    fn zzyy(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.y, self.y)
    }

    #[inline]
    fn zzyz(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.y, self.z)
    }

    #[inline]
    fn zzzx(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.z, self.x)
    }

    #[inline]
    fn zzzy(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.z, self.y)
    }

    #[inline]
    fn zzzz(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.z, self.z)
    }
}
