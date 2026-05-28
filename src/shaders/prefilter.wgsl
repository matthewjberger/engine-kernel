@group(0) @binding(0) var prefilter_output: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(1) var<uniform> prefilter_params: vec4<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(prefilter_output).x;
    if id.x >= size || id.y >= size {
        return;
    }
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + 0.5) / f32(size);
    let normal = cube_direction(id.z, uv);
    let roughness = prefilter_params.x;
    var color = vec3<f32>(0.0);
    var weight = 0.0;
    let samples = 128u;
    for (var index = 0u; index < samples; index = index + 1u) {
        let half_vector = importance_ggx(hammersley(index, samples), roughness, normal);
        let light = reflect(-normal, half_vector);
        let n_dot_l = dot(normal, light);
        if n_dot_l > 0.0 {
            color += environment(light) * n_dot_l;
            weight += n_dot_l;
        }
    }
    textureStore(prefilter_output, vec2<i32>(i32(id.x), i32(id.y)), i32(id.z), vec4<f32>(color / max(weight, 0.001), 1.0));
}
