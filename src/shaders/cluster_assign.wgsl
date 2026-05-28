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
struct LightGrid {
    offset: u32,
    count: u32,
}
struct ClusterLight {
    position: vec4<f32>,
    direction: vec4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: ClusterUniforms;
@group(0) @binding(1) var<storage, read> cluster_bounds: array<ClusterBounds>;
@group(0) @binding(2) var<storage, read_write> light_grid: array<LightGrid>;
@group(0) @binding(3) var<storage, read_write> light_indices: array<u32>;
@group(0) @binding(4) var<storage, read> lights: array<ClusterLight>;
fn sphere_aabb(center: vec3<f32>, radius: f32, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let closest = clamp(center, aabb_min, aabb_max);
    let offset = closest - center;
    return dot(offset, offset) <= radius * radius;
}
fn cone_aabb(tip: vec3<f32>, direction: vec3<f32>, range: f32, angle_cos: f32, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    if !sphere_aabb(tip, range, aabb_min, aabb_max) {
        return false;
    }
    let center = (aabb_min + aabb_max) * 0.5;
    let radius = length((aabb_max - aabb_min) * 0.5);
    let to_center = center - tip;
    let along = dot(to_center, direction);
    if along < 0.0 {
        return length(to_center) <= radius;
    }
    if along > range + radius {
        return false;
    }
    let on_axis = tip + direction * along;
    let perpendicular = length(center - on_axis);
    let cone_radius = along * sqrt(1.0 - angle_cos * angle_cos) / angle_cos;
    return perpendicular <= cone_radius + radius;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= uniforms.cluster_count.x || global_id.y >= uniforms.cluster_count.y || global_id.z >= uniforms.cluster_count.z {
        return;
    }
    let cluster_index = global_id.x + global_id.y * uniforms.cluster_count.x + global_id.z * uniforms.cluster_count.x * uniforms.cluster_count.y;
    let bounds = cluster_bounds[cluster_index];
    let aabb_min = bounds.min_point.xyz;
    let aabb_max = bounds.max_point.xyz;
    let base = cluster_index * uniforms.cluster_count.w;
    var count = 0u;
    for (var light_index = 0u; light_index < uniforms.num_lights; light_index = light_index + 1u) {
        let light = lights[light_index];
        let view_position = (uniforms.view * vec4<f32>(light.position.xyz, 1.0)).xyz;
        let range = light.position.w;
        var intersects = false;
        if range <= 0.0 {
            intersects = true;
        } else if light.direction.w < -1.5 {
            intersects = sphere_aabb(view_position, range, aabb_min, aabb_max);
        } else {
            let view_direction = normalize((uniforms.view * vec4<f32>(light.direction.xyz, 0.0)).xyz);
            intersects = cone_aabb(view_position, view_direction, range, light.direction.w, aabb_min, aabb_max);
        }
        if intersects && count < uniforms.cluster_count.w {
            light_indices[base + count] = light_index;
            count = count + 1u;
        }
    }
    light_grid[cluster_index].offset = base;
    light_grid[cluster_index].count = count;
}
