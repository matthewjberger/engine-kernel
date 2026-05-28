@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(scene_texture, scene_sampler, in.uv).rgb;
    return vec4<f32>(max(color - vec3<f32>(1.0), vec3<f32>(0.0)), 1.0);
}
