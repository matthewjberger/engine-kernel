struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}
struct CompactParams {
    batch_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<storage, read_write> commands: array<DrawIndirect>;
@group(0) @binding(1) var<storage, read_write> count: array<u32>;
@group(0) @binding(2) var<uniform> params: CompactParams;
@compute @workgroup_size(1)
fn main() {
    var dense = 0u;
    for (var i = 0u; i < params.batch_count; i++) {
        let command = commands[i];
        if command.instance_count > 0u {
            if dense != i {
                commands[dense] = command;
            }
            dense += 1u;
        }
    }
    count[0] = dense;
}
