struct ClusterUniforms {
    inverse_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    screen_size: vec2<f32>,
    z_near: f32,
    z_far: f32,
    cluster_count: vec4<u32>,
    tile_size: vec2<f32>,
    num_lights: u32,
    pad: u32,
}
struct ClusterBounds {
    min_point: vec4<f32>,
    max_point: vec4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: ClusterUniforms;
@group(0) @binding(1) var<storage, read_write> cluster_bounds: array<ClusterBounds>;
fn screen_to_view(screen_coord: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(
        (screen_coord.x / uniforms.screen_size.x) * 2.0 - 1.0,
        (1.0 - screen_coord.y / uniforms.screen_size.y) * 2.0 - 1.0,
        depth,
        1.0,
    );
    let view_position = uniforms.inverse_projection * ndc;
    return view_position.xyz / view_position.w;
}
fn line_intersection_with_z_plane(start: vec3<f32>, end: vec3<f32>, z: f32) -> vec3<f32> {
    let direction = end - start;
    let t = (z - start.z) / direction.z;
    return start + t * direction;
}
fn cluster_depth_to_view_z(slice: u32) -> f32 {
    let t = f32(slice) / f32(uniforms.cluster_count.z);
    return -uniforms.z_near * pow(uniforms.z_far / uniforms.z_near, t);
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= uniforms.cluster_count.x || global_id.y >= uniforms.cluster_count.y || global_id.z >= uniforms.cluster_count.z {
        return;
    }
    let tile_size = uniforms.tile_size;
    let min_screen = vec2<f32>(f32(global_id.x) * tile_size.x, f32(global_id.y) * tile_size.y);
    let max_screen = vec2<f32>(f32(global_id.x + 1u) * tile_size.x, f32(global_id.y + 1u) * tile_size.y);
    let eye = vec3<f32>(0.0, 0.0, 0.0);
    let min_view = screen_to_view(min_screen, 1.0);
    let max_view = screen_to_view(max_screen, 1.0);
    let near_z = cluster_depth_to_view_z(global_id.z);
    let far_z = cluster_depth_to_view_z(global_id.z + 1u);
    let min_near = line_intersection_with_z_plane(eye, min_view, near_z);
    let min_far = line_intersection_with_z_plane(eye, min_view, far_z);
    let max_near = line_intersection_with_z_plane(eye, max_view, near_z);
    let max_far = line_intersection_with_z_plane(eye, max_view, far_z);
    let aabb_min = min(min(min_near, min_far), min(max_near, max_far));
    let aabb_max = max(max(min_near, min_far), max(max_near, max_far));
    let cluster_index = global_id.x + global_id.y * uniforms.cluster_count.x + global_id.z * uniforms.cluster_count.x * uniforms.cluster_count.y;
    cluster_bounds[cluster_index].min_point = vec4<f32>(aabb_min, 0.0);
    cluster_bounds[cluster_index].max_point = vec4<f32>(aabb_max, 0.0);
}
