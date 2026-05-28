@group(0) @binding(0) var ao_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.uv * vec2<f32>(textureDimensions(ao_texture)));
    let center_depth = textureLoad(depth_texture, coord, 0);
    if (center_depth <= 0.0) {
        return vec4<f32>(1.0);
    }
    let center_normal = normalize(textureLoad(normal_texture, coord, 0).xyz);
    var total = 0.0;
    var weight_total = 0.0;
    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let sample_coord = coord + vec2<i32>(x, y);
            let sample_depth = textureLoad(depth_texture, sample_coord, 0);
            if (sample_depth <= 0.0) {
                continue;
            }
            let sample_normal = normalize(textureLoad(normal_texture, sample_coord, 0).xyz);
            let depth_weight = exp(-abs(sample_depth - center_depth) * 200.0);
            let normal_weight = pow(max(dot(sample_normal, center_normal), 0.0), 8.0);
            let weight = depth_weight * normal_weight;
            total += textureLoad(ao_texture, sample_coord, 0).r * weight;
            weight_total += weight;
        }
    }
    return vec4<f32>(vec3<f32>(total / max(weight_total, 0.0001)), 1.0);
}
