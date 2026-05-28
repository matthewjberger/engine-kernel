struct Camera {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec4<f32>,
    inverse_projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var equirect: texture_2d<f32>;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_direction: vec3<f32>,
}
@vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOutput {
    let position = vec4<f32>(f32(i32(index) / 2) * 4.0 - 1.0, f32(i32(index) & 1) * 4.0 - 1.0, 0.0, 1.0);
    let inverse_view_rotation = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let unprojected = camera.inverse_projection * position;
    var out: VertexOutput;
    out.position = position;
    out.world_direction = inverse_view_rotation * unprojected.xyz;
    return out;
}
struct SkyOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
}
@fragment fn fs(in: VertexOutput) -> SkyOutput {
    let d = normalize(in.world_direction);
    let uv = vec2<f32>(atan2(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
    let size = vec2<f32>(textureDimensions(equirect));
    let coord = vec2<i32>(clamp(uv * size, vec2<f32>(0.0), size - 1.0));
    var out: SkyOutput;
    out.color = vec4<f32>(textureLoad(equirect, coord, 0).rgb, 1.0);
    out.normal = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    return out;
}
