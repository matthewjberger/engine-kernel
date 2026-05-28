struct Exposure {
    target_log_luminance: f32,
    current_log_luminance: f32,
    adaptation_rate: f32,
    delta_time: f32,
    primed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(3) var ao_texture: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: vec4<f32>;
@group(0) @binding(5) var<storage, read> exposure: Exposure;
fn aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_texture, scene_sampler, in.uv).rgb;
    let bloom = textureSample(bloom_texture, scene_sampler, in.uv).rgb;
    let ao = textureSample(ao_texture, scene_sampler, in.uv).r;
    let exposure_scale = clamp(exp2(-exposure.current_log_luminance), 0.7, 1.6);
    return vec4<f32>(aces((scene * ao + bloom * params.x) * exposure_scale), 1.0);
}
