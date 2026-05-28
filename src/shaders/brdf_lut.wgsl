@group(0) @binding(0) var brdf_output: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(brdf_output);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let n_dot_v = (f32(id.x) + 0.5) / f32(size.x);
    let roughness = (f32(id.y) + 0.5) / f32(size.y);
    let view = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let normal = vec3<f32>(0.0, 0.0, 1.0);
    var a = 0.0;
    var b = 0.0;
    let samples = 512u;
    for (var index = 0u; index < samples; index = index + 1u) {
        let half_vector = importance_ggx(hammersley(index, samples), roughness, normal);
        let light = reflect(-view, half_vector);
        let n_dot_l = max(light.z, 0.0);
        let n_dot_h = max(half_vector.z, 0.0);
        let v_dot_h = max(dot(view, half_vector), 0.0);
        if n_dot_l > 0.0 {
            let g = geometry_ibl(n_dot_l, roughness) * geometry_ibl(n_dot_v, roughness);
            let g_vis = (g * v_dot_h) / max(n_dot_h * n_dot_v, 0.0001);
            let fc = pow(1.0 - v_dot_h, 5.0);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }
    textureStore(brdf_output, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(a / f32(samples), b / f32(samples), 0.0, 1.0));
}
