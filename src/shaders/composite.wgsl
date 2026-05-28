@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(3) var ao_texture: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: vec4<f32>;
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
    return vec4<f32>(aces(scene * ao + bloom * params.x), 1.0);
}
