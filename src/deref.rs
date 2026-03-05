#[repr(C)]
pub struct Cols3<V> {
    pub x_axis: V,
    pub y_axis: V,
    pub z_axis: V,
}

#[repr(C)]
pub struct Cols4<V> {
    pub x_axis: V,
    pub y_axis: V,
    pub z_axis: V,
    pub w_axis: V,
}
