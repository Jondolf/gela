use crate::{GVec2, GVec3, GVec4, Vec4Swizzles};

impl<T: Copy> Vec4Swizzles for GVec4<T> {
    type Vec2 = GVec2<T>;
    type Vec3 = GVec3<T>;

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
        Self::new(rhs.x, rhs.y, self.z, self.w)
    }

    #[inline]
    fn xz(self) -> GVec2<T> {
        GVec2::new(self.x, self.z)
    }

    #[inline]
    fn with_xz(self, rhs: GVec2<T>) -> Self {
        Self::new(rhs.x, self.y, rhs.y, self.w)
    }

    #[inline]
    fn xw(self) -> GVec2<T> {
        GVec2::new(self.x, self.w)
    }

    #[inline]
    fn with_xw(self, rhs: GVec2<T>) -> Self {
        Self::new(rhs.x, self.y, self.z, rhs.y)
    }

    #[inline]
    fn yx(self) -> GVec2<T> {
        GVec2::new(self.y, self.x)
    }

    #[inline]
    fn with_yx(self, rhs: GVec2<T>) -> Self {
        Self::new(rhs.y, rhs.x, self.z, self.w)
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
        Self::new(self.x, rhs.x, rhs.y, self.w)
    }

    #[inline]
    fn yw(self) -> GVec2<T> {
        GVec2::new(self.y, self.w)
    }

    #[inline]
    fn with_yw(self, rhs: GVec2<T>) -> Self {
        Self::new(self.x, rhs.x, self.z, rhs.y)
    }

    #[inline]
    fn zx(self) -> GVec2<T> {
        GVec2::new(self.z, self.x)
    }

    #[inline]
    fn with_zx(self, rhs: GVec2<T>) -> Self {
        Self::new(rhs.y, self.y, rhs.x, self.w)
    }

    #[inline]
    fn zy(self) -> GVec2<T> {
        GVec2::new(self.z, self.y)
    }

    #[inline]
    fn with_zy(self, rhs: GVec2<T>) -> Self {
        Self::new(self.x, rhs.y, rhs.x, self.w)
    }

    #[inline]
    fn zz(self) -> GVec2<T> {
        GVec2::new(self.z, self.z)
    }

    #[inline]
    fn zw(self) -> GVec2<T> {
        GVec2::new(self.z, self.w)
    }

    #[inline]
    fn with_zw(self, rhs: GVec2<T>) -> Self {
        Self::new(self.x, self.y, rhs.x, rhs.y)
    }

    #[inline]
    fn wx(self) -> GVec2<T> {
        GVec2::new(self.w, self.x)
    }

    #[inline]
    fn with_wx(self, rhs: GVec2<T>) -> Self {
        Self::new(rhs.y, self.y, self.z, rhs.x)
    }

    #[inline]
    fn wy(self) -> GVec2<T> {
        GVec2::new(self.w, self.y)
    }

    #[inline]
    fn with_wy(self, rhs: GVec2<T>) -> Self {
        Self::new(self.x, rhs.y, self.z, rhs.x)
    }

    #[inline]
    fn wz(self) -> GVec2<T> {
        GVec2::new(self.w, self.z)
    }

    #[inline]
    fn with_wz(self, rhs: GVec2<T>) -> Self {
        Self::new(self.x, self.y, rhs.y, rhs.x)
    }

    #[inline]
    fn ww(self) -> GVec2<T> {
        GVec2::new(self.w, self.w)
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
    fn xxw(self) -> GVec3<T> {
        GVec3::new(self.x, self.x, self.w)
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
    fn xyz(self) -> GVec3<T> {
        GVec3::new(self.x, self.y, self.z)
    }

    #[inline]
    fn with_xyz(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.x, rhs.y, rhs.z, self.w)
    }

    #[inline]
    fn xyw(self) -> GVec3<T> {
        GVec3::new(self.x, self.y, self.w)
    }

    #[inline]
    fn with_xyw(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.x, rhs.y, self.z, rhs.z)
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
    fn with_xzy(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.x, rhs.z, rhs.y, self.w)
    }

    #[inline]
    fn xzz(self) -> GVec3<T> {
        GVec3::new(self.x, self.z, self.z)
    }

    #[inline]
    fn xzw(self) -> GVec3<T> {
        GVec3::new(self.x, self.z, self.w)
    }

    #[inline]
    fn with_xzw(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.x, self.y, rhs.y, rhs.z)
    }

    #[inline]
    fn xwx(self) -> GVec3<T> {
        GVec3::new(self.x, self.w, self.x)
    }

    #[inline]
    fn xwy(self) -> GVec3<T> {
        GVec3::new(self.x, self.w, self.y)
    }

    #[inline]
    fn with_xwy(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.x, rhs.z, self.z, rhs.y)
    }

    #[inline]
    fn xwz(self) -> GVec3<T> {
        GVec3::new(self.x, self.w, self.z)
    }

    #[inline]
    fn with_xwz(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.x, self.y, rhs.z, rhs.y)
    }

    #[inline]
    fn xww(self) -> GVec3<T> {
        GVec3::new(self.x, self.w, self.w)
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
    fn with_yxz(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.y, rhs.x, rhs.z, self.w)
    }

    #[inline]
    fn yxw(self) -> GVec3<T> {
        GVec3::new(self.y, self.x, self.w)
    }

    #[inline]
    fn with_yxw(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.y, rhs.x, self.z, rhs.z)
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
    fn yyw(self) -> GVec3<T> {
        GVec3::new(self.y, self.y, self.w)
    }

    #[inline]
    fn yzx(self) -> GVec3<T> {
        GVec3::new(self.y, self.z, self.x)
    }

    #[inline]
    fn with_yzx(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.z, rhs.x, rhs.y, self.w)
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
    fn yzw(self) -> GVec3<T> {
        GVec3::new(self.y, self.z, self.w)
    }

    #[inline]
    fn with_yzw(self, rhs: GVec3<T>) -> Self {
        Self::new(self.x, rhs.x, rhs.y, rhs.z)
    }

    #[inline]
    fn ywx(self) -> GVec3<T> {
        GVec3::new(self.y, self.w, self.x)
    }

    #[inline]
    fn with_ywx(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.z, rhs.x, self.z, rhs.y)
    }

    #[inline]
    fn ywy(self) -> GVec3<T> {
        GVec3::new(self.y, self.w, self.y)
    }

    #[inline]
    fn ywz(self) -> GVec3<T> {
        GVec3::new(self.y, self.w, self.z)
    }

    #[inline]
    fn with_ywz(self, rhs: GVec3<T>) -> Self {
        Self::new(self.x, rhs.x, rhs.z, rhs.y)
    }

    #[inline]
    fn yww(self) -> GVec3<T> {
        GVec3::new(self.y, self.w, self.w)
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
    fn with_zxy(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.y, rhs.z, rhs.x, self.w)
    }

    #[inline]
    fn zxz(self) -> GVec3<T> {
        GVec3::new(self.z, self.x, self.z)
    }

    #[inline]
    fn zxw(self) -> GVec3<T> {
        GVec3::new(self.z, self.x, self.w)
    }

    #[inline]
    fn with_zxw(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.y, self.y, rhs.x, rhs.z)
    }

    #[inline]
    fn zyx(self) -> GVec3<T> {
        GVec3::new(self.z, self.y, self.x)
    }

    #[inline]
    fn with_zyx(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.z, rhs.y, rhs.x, self.w)
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
    fn zyw(self) -> GVec3<T> {
        GVec3::new(self.z, self.y, self.w)
    }

    #[inline]
    fn with_zyw(self, rhs: GVec3<T>) -> Self {
        Self::new(self.x, rhs.y, rhs.x, rhs.z)
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
    fn zzw(self) -> GVec3<T> {
        GVec3::new(self.z, self.z, self.w)
    }

    #[inline]
    fn zwx(self) -> GVec3<T> {
        GVec3::new(self.z, self.w, self.x)
    }

    #[inline]
    fn with_zwx(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.z, self.y, rhs.x, rhs.y)
    }

    #[inline]
    fn zwy(self) -> GVec3<T> {
        GVec3::new(self.z, self.w, self.y)
    }

    #[inline]
    fn with_zwy(self, rhs: GVec3<T>) -> Self {
        Self::new(self.x, rhs.z, rhs.x, rhs.y)
    }

    #[inline]
    fn zwz(self) -> GVec3<T> {
        GVec3::new(self.z, self.w, self.z)
    }

    #[inline]
    fn zww(self) -> GVec3<T> {
        GVec3::new(self.z, self.w, self.w)
    }

    #[inline]
    fn wxx(self) -> GVec3<T> {
        GVec3::new(self.w, self.x, self.x)
    }

    #[inline]
    fn wxy(self) -> GVec3<T> {
        GVec3::new(self.w, self.x, self.y)
    }

    #[inline]
    fn with_wxy(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.y, rhs.z, self.z, rhs.x)
    }

    #[inline]
    fn wxz(self) -> GVec3<T> {
        GVec3::new(self.w, self.x, self.z)
    }

    #[inline]
    fn with_wxz(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.y, self.y, rhs.z, rhs.x)
    }

    #[inline]
    fn wxw(self) -> GVec3<T> {
        GVec3::new(self.w, self.x, self.w)
    }

    #[inline]
    fn wyx(self) -> GVec3<T> {
        GVec3::new(self.w, self.y, self.x)
    }

    #[inline]
    fn with_wyx(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.z, rhs.y, self.z, rhs.x)
    }

    #[inline]
    fn wyy(self) -> GVec3<T> {
        GVec3::new(self.w, self.y, self.y)
    }

    #[inline]
    fn wyz(self) -> GVec3<T> {
        GVec3::new(self.w, self.y, self.z)
    }

    #[inline]
    fn with_wyz(self, rhs: GVec3<T>) -> Self {
        Self::new(self.x, rhs.y, rhs.z, rhs.x)
    }

    #[inline]
    fn wyw(self) -> GVec3<T> {
        GVec3::new(self.w, self.y, self.w)
    }

    #[inline]
    fn wzx(self) -> GVec3<T> {
        GVec3::new(self.w, self.z, self.x)
    }

    #[inline]
    fn with_wzx(self, rhs: GVec3<T>) -> Self {
        Self::new(rhs.z, self.y, rhs.y, rhs.x)
    }

    #[inline]
    fn wzy(self) -> GVec3<T> {
        GVec3::new(self.w, self.z, self.y)
    }

    #[inline]
    fn with_wzy(self, rhs: GVec3<T>) -> Self {
        Self::new(self.x, rhs.z, rhs.y, rhs.x)
    }

    #[inline]
    fn wzz(self) -> GVec3<T> {
        GVec3::new(self.w, self.z, self.z)
    }

    #[inline]
    fn wzw(self) -> GVec3<T> {
        GVec3::new(self.w, self.z, self.w)
    }

    #[inline]
    fn wwx(self) -> GVec3<T> {
        GVec3::new(self.w, self.w, self.x)
    }

    #[inline]
    fn wwy(self) -> GVec3<T> {
        GVec3::new(self.w, self.w, self.y)
    }

    #[inline]
    fn wwz(self) -> GVec3<T> {
        GVec3::new(self.w, self.w, self.z)
    }

    #[inline]
    fn www(self) -> GVec3<T> {
        GVec3::new(self.w, self.w, self.w)
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
    fn xxxw(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.x, self.w)
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
    fn xxyw(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.y, self.w)
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
    fn xxzw(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.z, self.w)
    }

    #[inline]
    fn xxwx(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.w, self.x)
    }

    #[inline]
    fn xxwy(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.w, self.y)
    }

    #[inline]
    fn xxwz(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.w, self.z)
    }

    #[inline]
    fn xxww(self) -> GVec4<T> {
        GVec4::new(self.x, self.x, self.w, self.w)
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
    fn xyxw(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.x, self.w)
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
    fn xyyw(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.y, self.w)
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
    fn xywx(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.w, self.x)
    }

    #[inline]
    fn xywy(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.w, self.y)
    }

    #[inline]
    fn xywz(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.w, self.z)
    }

    #[inline]
    fn xyww(self) -> GVec4<T> {
        GVec4::new(self.x, self.y, self.w, self.w)
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
    fn xzxw(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.x, self.w)
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
    fn xzyw(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.y, self.w)
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
    fn xzzw(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.z, self.w)
    }

    #[inline]
    fn xzwx(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.w, self.x)
    }

    #[inline]
    fn xzwy(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.w, self.y)
    }

    #[inline]
    fn xzwz(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.w, self.z)
    }

    #[inline]
    fn xzww(self) -> GVec4<T> {
        GVec4::new(self.x, self.z, self.w, self.w)
    }

    #[inline]
    fn xwxx(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.x, self.x)
    }

    #[inline]
    fn xwxy(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.x, self.y)
    }

    #[inline]
    fn xwxz(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.x, self.z)
    }

    #[inline]
    fn xwxw(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.x, self.w)
    }

    #[inline]
    fn xwyx(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.y, self.x)
    }

    #[inline]
    fn xwyy(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.y, self.y)
    }

    #[inline]
    fn xwyz(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.y, self.z)
    }

    #[inline]
    fn xwyw(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.y, self.w)
    }

    #[inline]
    fn xwzx(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.z, self.x)
    }

    #[inline]
    fn xwzy(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.z, self.y)
    }

    #[inline]
    fn xwzz(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.z, self.z)
    }

    #[inline]
    fn xwzw(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.z, self.w)
    }

    #[inline]
    fn xwwx(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.w, self.x)
    }

    #[inline]
    fn xwwy(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.w, self.y)
    }

    #[inline]
    fn xwwz(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.w, self.z)
    }

    #[inline]
    fn xwww(self) -> GVec4<T> {
        GVec4::new(self.x, self.w, self.w, self.w)
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
    fn yxxw(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.x, self.w)
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
    fn yxyw(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.y, self.w)
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
    fn yxzw(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.z, self.w)
    }

    #[inline]
    fn yxwx(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.w, self.x)
    }

    #[inline]
    fn yxwy(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.w, self.y)
    }

    #[inline]
    fn yxwz(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.w, self.z)
    }

    #[inline]
    fn yxww(self) -> GVec4<T> {
        GVec4::new(self.y, self.x, self.w, self.w)
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
    fn yyxw(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.x, self.w)
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
    fn yyyw(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.y, self.w)
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
    fn yyzw(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.z, self.w)
    }

    #[inline]
    fn yywx(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.w, self.x)
    }

    #[inline]
    fn yywy(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.w, self.y)
    }

    #[inline]
    fn yywz(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.w, self.z)
    }

    #[inline]
    fn yyww(self) -> GVec4<T> {
        GVec4::new(self.y, self.y, self.w, self.w)
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
    fn yzxw(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.x, self.w)
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
    fn yzyw(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.y, self.w)
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
    fn yzzw(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.z, self.w)
    }

    #[inline]
    fn yzwx(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.w, self.x)
    }

    #[inline]
    fn yzwy(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.w, self.y)
    }

    #[inline]
    fn yzwz(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.w, self.z)
    }

    #[inline]
    fn yzww(self) -> GVec4<T> {
        GVec4::new(self.y, self.z, self.w, self.w)
    }

    #[inline]
    fn ywxx(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.x, self.x)
    }

    #[inline]
    fn ywxy(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.x, self.y)
    }

    #[inline]
    fn ywxz(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.x, self.z)
    }

    #[inline]
    fn ywxw(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.x, self.w)
    }

    #[inline]
    fn ywyx(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.y, self.x)
    }

    #[inline]
    fn ywyy(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.y, self.y)
    }

    #[inline]
    fn ywyz(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.y, self.z)
    }

    #[inline]
    fn ywyw(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.y, self.w)
    }

    #[inline]
    fn ywzx(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.z, self.x)
    }

    #[inline]
    fn ywzy(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.z, self.y)
    }

    #[inline]
    fn ywzz(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.z, self.z)
    }

    #[inline]
    fn ywzw(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.z, self.w)
    }

    #[inline]
    fn ywwx(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.w, self.x)
    }

    #[inline]
    fn ywwy(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.w, self.y)
    }

    #[inline]
    fn ywwz(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.w, self.z)
    }

    #[inline]
    fn ywww(self) -> GVec4<T> {
        GVec4::new(self.y, self.w, self.w, self.w)
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
    fn zxxw(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.x, self.w)
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
    fn zxyw(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.y, self.w)
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
    fn zxzw(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.z, self.w)
    }

    #[inline]
    fn zxwx(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.w, self.x)
    }

    #[inline]
    fn zxwy(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.w, self.y)
    }

    #[inline]
    fn zxwz(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.w, self.z)
    }

    #[inline]
    fn zxww(self) -> GVec4<T> {
        GVec4::new(self.z, self.x, self.w, self.w)
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
    fn zyxw(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.x, self.w)
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
    fn zyyw(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.y, self.w)
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
    fn zyzw(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.z, self.w)
    }

    #[inline]
    fn zywx(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.w, self.x)
    }

    #[inline]
    fn zywy(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.w, self.y)
    }

    #[inline]
    fn zywz(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.w, self.z)
    }

    #[inline]
    fn zyww(self) -> GVec4<T> {
        GVec4::new(self.z, self.y, self.w, self.w)
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
    fn zzxw(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.x, self.w)
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
    fn zzyw(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.y, self.w)
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

    #[inline]
    fn zzzw(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.z, self.w)
    }

    #[inline]
    fn zzwx(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.w, self.x)
    }

    #[inline]
    fn zzwy(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.w, self.y)
    }

    #[inline]
    fn zzwz(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.w, self.z)
    }

    #[inline]
    fn zzww(self) -> GVec4<T> {
        GVec4::new(self.z, self.z, self.w, self.w)
    }

    #[inline]
    fn zwxx(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.x, self.x)
    }

    #[inline]
    fn zwxy(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.x, self.y)
    }

    #[inline]
    fn zwxz(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.x, self.z)
    }

    #[inline]
    fn zwxw(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.x, self.w)
    }

    #[inline]
    fn zwyx(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.y, self.x)
    }

    #[inline]
    fn zwyy(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.y, self.y)
    }

    #[inline]
    fn zwyz(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.y, self.z)
    }

    #[inline]
    fn zwyw(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.y, self.w)
    }

    #[inline]
    fn zwzx(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.z, self.x)
    }

    #[inline]
    fn zwzy(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.z, self.y)
    }

    #[inline]
    fn zwzz(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.z, self.z)
    }

    #[inline]
    fn zwzw(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.z, self.w)
    }

    #[inline]
    fn zwwx(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.w, self.x)
    }

    #[inline]
    fn zwwy(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.w, self.y)
    }

    #[inline]
    fn zwwz(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.w, self.z)
    }

    #[inline]
    fn zwww(self) -> GVec4<T> {
        GVec4::new(self.z, self.w, self.w, self.w)
    }

    #[inline]
    fn wxxx(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.x, self.x)
    }

    #[inline]
    fn wxxy(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.x, self.y)
    }

    #[inline]
    fn wxxz(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.x, self.z)
    }

    #[inline]
    fn wxxw(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.x, self.w)
    }

    #[inline]
    fn wxyx(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.y, self.x)
    }

    #[inline]
    fn wxyy(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.y, self.y)
    }

    #[inline]
    fn wxyz(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.y, self.z)
    }

    #[inline]
    fn wxyw(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.y, self.w)
    }

    #[inline]
    fn wxzx(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.z, self.x)
    }

    #[inline]
    fn wxzy(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.z, self.y)
    }

    #[inline]
    fn wxzz(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.z, self.z)
    }

    #[inline]
    fn wxzw(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.z, self.w)
    }

    #[inline]
    fn wxwx(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.w, self.x)
    }

    #[inline]
    fn wxwy(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.w, self.y)
    }

    #[inline]
    fn wxwz(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.w, self.z)
    }

    #[inline]
    fn wxww(self) -> GVec4<T> {
        GVec4::new(self.w, self.x, self.w, self.w)
    }

    #[inline]
    fn wyxx(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.x, self.x)
    }

    #[inline]
    fn wyxy(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.x, self.y)
    }

    #[inline]
    fn wyxz(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.x, self.z)
    }

    #[inline]
    fn wyxw(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.x, self.w)
    }

    #[inline]
    fn wyyx(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.y, self.x)
    }

    #[inline]
    fn wyyy(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.y, self.y)
    }

    #[inline]
    fn wyyz(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.y, self.z)
    }

    #[inline]
    fn wyyw(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.y, self.w)
    }

    #[inline]
    fn wyzx(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.z, self.x)
    }

    #[inline]
    fn wyzy(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.z, self.y)
    }

    #[inline]
    fn wyzz(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.z, self.z)
    }

    #[inline]
    fn wyzw(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.z, self.w)
    }

    #[inline]
    fn wywx(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.w, self.x)
    }

    #[inline]
    fn wywy(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.w, self.y)
    }

    #[inline]
    fn wywz(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.w, self.z)
    }

    #[inline]
    fn wyww(self) -> GVec4<T> {
        GVec4::new(self.w, self.y, self.w, self.w)
    }

    #[inline]
    fn wzxx(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.x, self.x)
    }

    #[inline]
    fn wzxy(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.x, self.y)
    }

    #[inline]
    fn wzxz(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.x, self.z)
    }

    #[inline]
    fn wzxw(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.x, self.w)
    }

    #[inline]
    fn wzyx(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.y, self.x)
    }

    #[inline]
    fn wzyy(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.y, self.y)
    }

    #[inline]
    fn wzyz(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.y, self.z)
    }

    #[inline]
    fn wzyw(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.y, self.w)
    }

    #[inline]
    fn wzzx(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.z, self.x)
    }

    #[inline]
    fn wzzy(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.z, self.y)
    }

    #[inline]
    fn wzzz(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.z, self.z)
    }

    #[inline]
    fn wzzw(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.z, self.w)
    }

    #[inline]
    fn wzwx(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.w, self.x)
    }

    #[inline]
    fn wzwy(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.w, self.y)
    }

    #[inline]
    fn wzwz(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.w, self.z)
    }

    #[inline]
    fn wzww(self) -> GVec4<T> {
        GVec4::new(self.w, self.z, self.w, self.w)
    }

    #[inline]
    fn wwxx(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.x, self.x)
    }

    #[inline]
    fn wwxy(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.x, self.y)
    }

    #[inline]
    fn wwxz(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.x, self.z)
    }

    #[inline]
    fn wwxw(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.x, self.w)
    }

    #[inline]
    fn wwyx(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.y, self.x)
    }

    #[inline]
    fn wwyy(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.y, self.y)
    }

    #[inline]
    fn wwyz(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.y, self.z)
    }

    #[inline]
    fn wwyw(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.y, self.w)
    }

    #[inline]
    fn wwzx(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.z, self.x)
    }

    #[inline]
    fn wwzy(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.z, self.y)
    }

    #[inline]
    fn wwzz(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.z, self.z)
    }

    #[inline]
    fn wwzw(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.z, self.w)
    }

    #[inline]
    fn wwwx(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.w, self.x)
    }

    #[inline]
    fn wwwy(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.w, self.y)
    }

    #[inline]
    fn wwwz(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.w, self.z)
    }

    #[inline]
    fn wwww(self) -> GVec4<T> {
        GVec4::new(self.w, self.w, self.w, self.w)
    }
}
