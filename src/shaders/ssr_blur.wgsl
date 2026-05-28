struct SsrBlurParams {
    screen_size: vec2<f32>,
    depth_threshold: f32,
    normal_threshold: f32,
}
@group(0) @binding(0) var ssr_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var ssr_sampler: sampler;
@group(0) @binding(4) var<uniform> params: SsrBlurParams;
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel_size = 1.0 / params.screen_size;
    let pixel = vec2<i32>(in.uv * params.screen_size);
    let center_depth = textureLoad(depth_texture, pixel, 0);
    if center_depth == 0.0 {
        return vec4<f32>(0.0);
    }
    let center_normal = textureLoad(normal_texture, pixel, 0).xyz;
    var result = vec4<f32>(0.0);
    var total_weight = 0.0;
    for (var x = -3i; x <= 3i; x++) {
        for (var y = -3i; y <= 3i; y++) {
            let sample_uv = in.uv + vec2<f32>(f32(x), f32(y)) * texel_size;
            if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
                continue;
            }
            let sample_pixel = vec2<i32>(sample_uv * params.screen_size);
            let sample_color = textureSampleLevel(ssr_texture, ssr_sampler, sample_uv, 0.0);
            let sample_depth = textureLoad(depth_texture, sample_pixel, 0);
            let sample_normal = textureLoad(normal_texture, sample_pixel, 0).xyz;
            let depth_weight = 1.0 - clamp(abs(center_depth - sample_depth) / params.depth_threshold, 0.0, 1.0);
            let normal_weight = pow(max(dot(center_normal, sample_normal), 0.0), params.normal_threshold);
            let spatial_dist = length(vec2<f32>(f32(x), f32(y)));
            let spatial_weight = exp(-spatial_dist * spatial_dist / 12.0);
            let weight = depth_weight * normal_weight * spatial_weight;
            result += sample_color * weight;
            total_weight += weight;
        }
    }
    if total_weight > 0.0 {
        result /= total_weight;
    }
    return result;
}
