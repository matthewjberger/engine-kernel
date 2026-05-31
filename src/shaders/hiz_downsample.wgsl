@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var destination: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_size = params.xy;
    if id.x >= dst_size.x || id.y >= dst_size.y {
        return;
    }
    let src = vec2<i32>(i32(id.x) * 2, i32(id.y) * 2);
    let src_size = vec2<i32>(textureDimensions(source, 0));
    let x1 = min(src.x + 1, src_size.x - 1);
    let y1 = min(src.y + 1, src_size.y - 1);
    let d00 = textureLoad(source, vec2<i32>(src.x, src.y), 0).r;
    let d10 = textureLoad(source, vec2<i32>(x1, src.y), 0).r;
    let d01 = textureLoad(source, vec2<i32>(src.x, y1), 0).r;
    let d11 = textureLoad(source, vec2<i32>(x1, y1), 0).r;
    let reduced = min(min(d00, d10), min(d01, d11));
    textureStore(destination, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(reduced, 0.0, 0.0, 1.0));
}
