struct RestVertex {
    px: f32,
    py: f32,
    pz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    u: f32,
    v: f32,
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
    j0: u32,
    j1: u32,
    j2: u32,
    j3: u32,
}
struct OutVertex {
    px: f32,
    py: f32,
    pz: f32,
    cr: f32,
    cg: f32,
    cb: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    u: f32,
    v: f32,
}
@group(0) @binding(0) var<storage, read> rest: array<RestVertex>;
@group(0) @binding(1) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read_write> skinned: array<OutVertex>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= params.x {
        return;
    }
    let vertex = rest[index];
    let position = vec3<f32>(vertex.px, vertex.py, vertex.pz);
    let normal = vec3<f32>(vertex.nx, vertex.ny, vertex.nz);
    let joint_offset = params.y;
    var joints = array<u32, 4>(vertex.j0, vertex.j1, vertex.j2, vertex.j3);
    var weights = array<f32, 4>(vertex.w0, vertex.w1, vertex.w2, vertex.w3);
    var skinned_position = vec3<f32>(0.0);
    var skinned_normal = vec3<f32>(0.0);
    for (var slot = 0u; slot < 4u; slot = slot + 1u) {
        let weight = weights[slot];
        if weight > 0.0 {
            let joint_matrix = joint_matrices[joint_offset + joints[slot]];
            skinned_position = skinned_position + (joint_matrix * vec4<f32>(position, 1.0)).xyz * weight;
            let normal_matrix = mat3x3<f32>(
                joint_matrix[0].xyz,
                joint_matrix[1].xyz,
                joint_matrix[2].xyz,
            );
            skinned_normal = skinned_normal + (normal_matrix * normal) * weight;
        }
    }
    skinned_normal = normalize(skinned_normal);
    var out: OutVertex;
    out.px = skinned_position.x;
    out.py = skinned_position.y;
    out.pz = skinned_position.z;
    out.cr = 1.0;
    out.cg = 1.0;
    out.cb = 1.0;
    out.nx = skinned_normal.x;
    out.ny = skinned_normal.y;
    out.nz = skinned_normal.z;
    out.u = vertex.u;
    out.v = vertex.v;
    skinned[index] = out;
}
