struct Camera {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec4<f32>,
    inverse_projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var environment: texture_cube<f32>;
@group(0) @binding(2) var environment_sampler: sampler;
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
    let color = textureSampleLevel(environment, environment_sampler, normalize(in.world_direction), 0.0).rgb;
    var out: SkyOutput;
    out.color = vec4<f32>(color, 1.0);
    out.normal = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    return out;
}
