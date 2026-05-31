@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var destination: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.x || id.y >= params.y {
        return;
    }
    let depth = textureLoad(depth_texture, vec2<i32>(i32(id.x), i32(id.y)), 0);
    textureStore(destination, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(depth, 0.0, 0.0, 1.0));
}
