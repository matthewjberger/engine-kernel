struct Ssao {
    inverse_projection: mat4x4<f32>,
    params: vec4<f32>,
}
@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> data: Ssao;
fn view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let position = data.inverse_projection * vec4<f32>(ndc, 1.0);
    return position.xyz / position.w;
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(depth_texture));
    let coord = vec2<i32>(in.uv * dimensions);
    let depth = textureLoad(depth_texture, coord, 0);
    if (depth <= 0.0) {
        return vec4<f32>(1.0);
    }
    let position = view_position(in.uv, depth);
    let normal = normalize(textureLoad(normal_texture, coord, 0).xyz);
    let radius = data.params.x;
    let bias = data.params.y;
    let strength = data.params.z;
    var occlusion = 0.0;
    for (var index = 0; index < 8; index = index + 1) {
        let angle = f32(index) / 8.0 * 6.2831853;
        let sample_coord = coord + vec2<i32>(vec2<f32>(cos(angle), sin(angle)) * radius);
        let sample_depth = textureLoad(depth_texture, sample_coord, 0);
        if (sample_depth <= 0.0) {
            continue;
        }
        let sample_uv = (vec2<f32>(sample_coord) + 0.5) / dimensions;
        let difference = view_position(sample_uv, sample_depth) - position;
        let range = 1.0 / (1.0 + dot(difference, difference));
        occlusion += max(dot(normalize(difference), normal) - bias, 0.0) * range;
    }
    let ao = clamp(1.0 - occlusion / 8.0 * strength, 0.0, 1.0);
    return vec4<f32>(vec3<f32>(ao), 1.0);
}
