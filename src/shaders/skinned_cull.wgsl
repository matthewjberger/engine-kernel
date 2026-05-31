struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}
struct CullUniform {
    frustum: array<vec4<f32>, 6>,
    sphere: vec4<f32>,
    vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<storage, read_write> command: array<DrawIndirect>;
@group(0) @binding(1) var<uniform> cull: CullUniform;
@compute @workgroup_size(1)
fn main() {
    var visible = true;
    for (var plane = 0u; plane < 6u; plane += 1u) {
        if dot(cull.frustum[plane].xyz, cull.sphere.xyz) + cull.frustum[plane].w < -cull.sphere.w {
            visible = false;
            break;
        }
    }
    command[0] = DrawIndirect(cull.vertex_count, u32(visible), 0u, 0u);
}
