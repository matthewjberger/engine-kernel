@group(0) @binding(0) var irradiance_output: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(irradiance_output).x;
    if id.x >= size || id.y >= size {
        return;
    }
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + 0.5) / f32(size);
    let normal = cube_direction(id.z, uv);
    let basis = tangent_basis(normal);
    var color = vec3<f32>(0.0);
    let samples = 256u;
    for (var index = 0u; index < samples; index = index + 1u) {
        let xi = hammersley(index, samples);
        let phi = 2.0 * PI * xi.x;
        let cos_theta = sqrt(1.0 - xi.y);
        let sin_theta = sqrt(xi.y);
        let local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        color += environment(basis * local);
    }
    textureStore(irradiance_output, vec2<i32>(i32(id.x), i32(id.y)), i32(id.z), vec4<f32>(color / f32(samples), 1.0));
}
