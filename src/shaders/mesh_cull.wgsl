struct Instance {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    m3: vec4<f32>,
    emissive_roughness: vec4<f32>,
    albedo_metallic: vec4<f32>,
    layers: vec4<f32>,
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
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<storage, read> source: array<Instance>;
@group(0) @binding(1) var<storage, read_write> visible_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect: DrawIndirect;
@group(0) @binding(3) var<uniform> bounds: Bounds;
@group(0) @binding(4) var<uniform> cull: CullUniform;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= cull.count {
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
    if visible {
        let write_index = atomicAdd(&indirect.instance_count, 1u);
        visible_indices[write_index] = index;
    }
}
