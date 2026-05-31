struct Instance {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    m3: vec4<f32>,
    normal_matrix: mat3x3<f32>,
    emissive_roughness: vec4<f32>,
    albedo_metallic: vec4<f32>,
    layers: vec4<f32>,
    visible: vec4<u32>,
}
struct DrawIndirect {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}
struct Bounds {
    center: vec3<f32>,
    radius: f32,
}
struct CullUniform {
    frustum: array<vec4<f32>, 6>,
    prev_view_projection: mat4x4<f32>,
    screen_size: vec2<f32>,
    count: u32,
    hiz_mip_count: u32,
    occlusion_enabled: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<storage, read_write> source: array<Instance>;
@group(0) @binding(1) var<storage, read_write> visible_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect: DrawIndirect;
@group(0) @binding(3) var<uniform> bounds: Bounds;
@group(0) @binding(4) var<uniform> cull: CullUniform;
@group(0) @binding(5) var hiz_texture: texture_2d<f32>;
fn compute_normal_matrix(model: mat4x4<f32>) -> mat3x3<f32> {
    let a = model[0].xyz;
    let b = model[1].xyz;
    let c = model[2].xyz;
    let cofactor_0 = cross(b, c);
    let cofactor_1 = cross(c, a);
    let cofactor_2 = cross(a, b);
    let determinant = dot(a, cofactor_0);
    if abs(determinant) < 1e-8 {
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    let inverse_determinant = 1.0 / determinant;
    return mat3x3<f32>(
        cofactor_0 * inverse_determinant,
        cofactor_1 * inverse_determinant,
        cofactor_2 * inverse_determinant,
    );
}
fn is_occluded(corners: array<vec3<f32>, 8>) -> bool {
    var screen_min = vec2<f32>(1.0);
    var screen_max = vec2<f32>(0.0);
    var nearest_z = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let clip = cull.prev_view_projection * vec4<f32>(corners[i], 1.0);
        if clip.w <= 0.0 {
            return false;
        }
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
        screen_min = min(screen_min, uv);
        screen_max = max(screen_max, uv);
        nearest_z = max(nearest_z, ndc.z);
    }
    let padded_min = clamp(screen_min, vec2<f32>(0.0), vec2<f32>(1.0));
    let padded_max = clamp(screen_max, vec2<f32>(0.0), vec2<f32>(1.0));
    let rect_size = max(
        (padded_max.x - padded_min.x) * cull.screen_size.x,
        (padded_max.y - padded_min.y) * cull.screen_size.y,
    );
    let mip = i32(clamp(ceil(log2(max(rect_size, 1.0))), 0.0, f32(cull.hiz_mip_count) - 1.0));
    let mip_size = vec2<f32>(textureDimensions(hiz_texture, mip));
    let min_texel = vec2<i32>(padded_min * mip_size);
    let max_texel = vec2<i32>(padded_max * mip_size);
    var hiz_depth = 1.0;
    for (var y = min_texel.y; y <= max_texel.y; y = y + 1) {
        for (var x = min_texel.x; x <= max_texel.x; x = x + 1) {
            let d = textureLoad(hiz_texture, vec2<i32>(x, y), mip).r;
            hiz_depth = min(hiz_depth, d);
        }
    }
    return nearest_z < hiz_depth * 0.98;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= cull.count {
        return;
    }
    if source[index].visible.x == 0u {
        return;
    }
    let instance = source[index];
    let model = mat4x4<f32>(instance.m0, instance.m1, instance.m2, instance.m3);
    let world_center = (model * vec4<f32>(bounds.center, 1.0)).xyz;
    let scale_x = length(model[0].xyz);
    let scale_y = length(model[1].xyz);
    let scale_z = length(model[2].xyz);
    let radius = max(max(scale_x, scale_y), scale_z) * bounds.radius;
    var visible = true;
    for (var plane = 0u; plane < 6u; plane += 1u) {
        if dot(cull.frustum[plane].xyz, world_center) + cull.frustum[plane].w < -radius {
            visible = false;
            break;
        }
    }
    if visible && cull.occlusion_enabled != 0u {
        let lo = bounds.center - vec3<f32>(bounds.radius);
        let hi = bounds.center + vec3<f32>(bounds.radius);
        let edge0 = model[0].xyz * (hi.x - lo.x);
        let edge1 = model[1].xyz * (hi.y - lo.y);
        let edge2 = model[2].xyz * (hi.z - lo.z);
        let base = (model * vec4<f32>(lo, 1.0)).xyz;
        let corners = array<vec3<f32>, 8>(
            base,
            base + edge0,
            base + edge1,
            base + edge0 + edge1,
            base + edge2,
            base + edge0 + edge2,
            base + edge1 + edge2,
            base + edge0 + edge1 + edge2,
        );
        if is_occluded(corners) {
            visible = false;
        }
    }
    if visible {
        source[index].normal_matrix = compute_normal_matrix(model);
        let write_index = atomicAdd(&indirect.instance_count, 1u);
        visible_indices[write_index] = index;
    }
}
