const PI: f32 = 3.14159265;
@group(0) @binding(2) var equirect: texture_2d<f32>;
fn environment(direction: vec3<f32>) -> vec3<f32> {
    let d = normalize(direction);
    let uv = vec2<f32>(atan2(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
    let size = vec2<f32>(textureDimensions(equirect));
    let coord = vec2<i32>(clamp(uv * size, vec2<f32>(0.0), size - 1.0));
    return textureLoad(equirect, coord, 0).rgb;
}
fn cube_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let s = uv * 2.0 - 1.0;
    var dir = vec3<f32>(-s.x, -s.y, -1.0);
    switch face {
        case 0u { dir = vec3<f32>(1.0, -s.y, -s.x); }
        case 1u { dir = vec3<f32>(-1.0, -s.y, s.x); }
        case 2u { dir = vec3<f32>(s.x, 1.0, s.y); }
        case 3u { dir = vec3<f32>(s.x, -1.0, -s.y); }
        case 4u { dir = vec3<f32>(s.x, -s.y, 1.0); }
        default { dir = vec3<f32>(-s.x, -s.y, -1.0); }
    }
    return normalize(dir);
}
fn radical_inverse(value: u32) -> f32 {
    var bits = value;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}
fn hammersley(index: u32, count: u32) -> vec2<f32> {
    return vec2<f32>(f32(index) / f32(count), radical_inverse(index));
}
fn tangent_basis(normal: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.z) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}
fn importance_ggx(xi: vec2<f32>, roughness: f32, normal: vec3<f32>) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let half_local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return tangent_basis(normal) * half_local;
}
fn geometry_ibl(n_dot: f32, roughness: f32) -> f32 {
    let k = roughness * roughness / 2.0;
    return n_dot / (n_dot * (1.0 - k) + k);
}
