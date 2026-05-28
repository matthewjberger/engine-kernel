struct Camera {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec4<f32>,
    inverse_projection: mat4x4<f32>,
}
struct Lights {
    ambient: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    point_count: vec4<f32>,
    point_position: array<vec4<f32>, 8>,
    point_color: array<vec4<f32>, 8>,
    point_direction: array<vec4<f32>, 8>,
    point_shadow: array<vec4<f32>, 8>,
}
struct Shadow {
    cascade_view_projection: array<mat4x4<f32>, 4>,
    split_distances: vec4<f32>,
    atlas_offset: array<vec4<f32>, 4>,
    atlas_scale: vec4<f32>,
    params: vec4<f32>,
}
struct SpotShadow {
    view_projection: array<mat4x4<f32>, 4>,
    atlas_rect: array<vec4<f32>, 4>,
}
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
struct LightGrid {
    offset: u32,
    count: u32,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lights: Lights;
@group(0) @binding(2) var shadow_texture: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler;
@group(0) @binding(4) var<uniform> shadow: Shadow;
@group(0) @binding(5) var spotlight_atlas: texture_depth_2d;
@group(0) @binding(6) var<uniform> spot_shadow: SpotShadow;
@group(0) @binding(7) var<uniform> cluster: ClusterUniforms;
@group(0) @binding(8) var<storage, read> light_grid: array<LightGrid>;
@group(0) @binding(9) var<storage, read> light_indices: array<u32>;
@group(0) @binding(10) var albedo_textures: texture_2d_array<f32>;
@group(0) @binding(11) var albedo_sampler: sampler;
@group(0) @binding(12) var irradiance_map: texture_cube<f32>;
@group(0) @binding(13) var prefiltered_map: texture_cube<f32>;
@group(0) @binding(14) var brdf_lut: texture_2d<f32>;
@group(0) @binding(15) var ibl_sampler: sampler;
@group(0) @binding(16) var normal_textures: texture_2d_array<f32>;
@group(0) @binding(17) var orm_textures: texture_2d_array<f32>;
@group(0) @binding(18) var emissive_textures: texture_2d_array<f32>;
@group(0) @binding(19) var point_cube: texture_cube<f32>;
@group(0) @binding(20) var<uniform> point_cube_params: vec4<f32>;

fn point_shadow_factor(index: u32, world_position: vec3<f32>) -> f32 {
    if i32(index) != i32(point_cube_params.x) {
        return 1.0;
    }
    let to_fragment = world_position - lights.point_position[index].xyz;
    let distance = length(to_fragment) / lights.point_position[index].w;
    let sampled = textureSampleLevel(point_cube, ibl_sampler, normalize(to_fragment), 0.0).r;
    return select(0.0, 1.0, distance - point_cube_params.y <= sampled);
}

fn cotangent_frame(normal: vec3<f32>, position: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
    let dp1 = dpdx(position);
    let dp2 = dpdy(position);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, normal);
    let dp1perp = cross(normal, dp1);
    let tangent = dp2perp * duv1.x + dp1perp * duv2.x;
    let bitangent = dp2perp * duv1.y + dp1perp * duv2.y;
    let inverse_max = inverseSqrt(max(dot(tangent, tangent), dot(bitangent, bitangent)));
    return mat3x3<f32>(tangent * inverse_max, bitangent * inverse_max, normal);
}

fn get_cluster_index(frag_coord: vec2<f32>, view_depth: f32) -> u32 {
    let tile_x = u32(frag_coord.x / cluster.tile_size.x);
    let tile_y = u32(frag_coord.y / cluster.tile_size.y);
    let log_ratio = log(cluster.z_far / cluster.z_near);
    let safe_depth = max(view_depth, cluster.z_near);
    let slice = u32(log(safe_depth / cluster.z_near) / log_ratio * f32(cluster.cluster_count.z));
    let clamped_slice = clamp(slice, 0u, cluster.cluster_count.z - 1u);
    let clamped_x = clamp(tile_x, 0u, cluster.cluster_count.x - 1u);
    let clamped_y = clamp(tile_y, 0u, cluster.cluster_count.y - 1u);
    return clamped_x + clamped_y * cluster.cluster_count.x + clamped_slice * cluster.cluster_count.x * cluster.cluster_count.y;
}

const POISSON_8: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-0.7071, -0.7071),
    vec2<f32>(0.7071, -0.7071),
    vec2<f32>(-0.7071, 0.7071),
    vec2<f32>(0.7071, 0.7071),
    vec2<f32>(-1.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, -1.0),
    vec2<f32>(0.0, 1.0),
);
const POISSON_16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>(0.94558609, -0.76890725),
    vec2<f32>(-0.094184101, -0.92938870),
    vec2<f32>(0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543, 0.27676845),
    vec2<f32>(0.97484398, 0.75648379),
    vec2<f32>(0.44323325, -0.97511554),
    vec2<f32>(0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>(0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507),
    vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>(0.19984126, 0.78641367),
    vec2<f32>(0.14383161, -0.14100790),
);

fn select_cascade(view_depth: f32) -> i32 {
    for (var index = 0; index < 4; index = index + 1) {
        if view_depth < shadow.split_distances[index] {
            return index;
        }
    }
    return 3;
}

fn sample_cascade(world_position: vec3<f32>, world_normal: vec3<f32>, cascade: i32) -> f32 {
    let texel_world = shadow.atlas_offset[cascade].z;
    let offset_position = world_position + world_normal * shadow.params.y * texel_world;
    let clip = shadow.cascade_view_projection[cascade] * vec4<f32>(offset_position, 1.0);
    let ndc = clip.xyz / clip.w;
    let scale = shadow.atlas_scale.x;
    let atlas_offset = shadow.atlas_offset[cascade].xy;
    let uv = (ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5)) * scale + atlas_offset;
    let depth = ndc.z + shadow.params.z;
    let slot_min = atlas_offset;
    let slot_max = atlas_offset + scale;
    let in_bounds = uv.x >= slot_min.x && uv.x <= slot_max.x && uv.y >= slot_min.y && uv.y <= slot_max.y && ndc.z >= 0.0 && ndc.z <= 1.0;
    if !in_bounds {
        return 1.0;
    }
    let texel = 1.0 / 2048.0;
    let safe_min = slot_min + scale * 0.01;
    let safe_max = slot_max - scale * 0.01;
    let light_size = shadow.params.x;
    var blocker_sum = 0.0;
    var blockers = 0u;
    for (var index = 0; index < 8; index = index + 1) {
        let sample_uv = clamp(uv + POISSON_8[index] * texel * light_size * 10.0, safe_min, safe_max);
        let sampled = textureSampleLevel(shadow_texture, shadow_sampler, sample_uv, 0);
        if sampled > depth {
            blocker_sum += sampled;
            blockers += 1u;
        }
    }
    if blockers == 0u {
        return 1.0;
    }
    let average = blocker_sum / f32(blockers);
    let penumbra = (average - depth) / max(1.0 - average, 0.001);
    let filter_radius = clamp(penumbra * light_size * 20.0, 1.0, 8.0);
    var visibility = 0.0;
    for (var index = 0; index < 16; index = index + 1) {
        let sample_uv = clamp(uv + POISSON_16[index] * texel * filter_radius, safe_min, safe_max);
        let sampled = textureSampleLevel(shadow_texture, shadow_sampler, sample_uv, 0);
        visibility += select(0.0, 1.0, depth >= sampled);
    }
    return visibility / 16.0;
}

fn calculate_shadow(world_position: vec3<f32>, world_normal: vec3<f32>, view_depth: f32) -> f32 {
    if shadow.params.w < 0.5 {
        return 1.0;
    }
    let cascade = select_cascade(view_depth);
    let factor = sample_cascade(world_position, world_normal, cascade);
    if cascade < 3 {
        let cascade_end = shadow.split_distances[cascade];
        let blend_range = cascade_end * 0.25;
        let blend_start = cascade_end - blend_range;
        if view_depth > blend_start {
            let blend = (view_depth - blend_start) / blend_range;
            return mix(factor, sample_cascade(world_position, world_normal, cascade + 1), blend);
        }
    }
    return factor;
}

fn spot_shadow_factor(world_position: vec3<f32>, shadow_index: i32) -> f32 {
    if shadow_index < 0 {
        return 1.0;
    }
    let clip = spot_shadow.view_projection[shadow_index] * vec4<f32>(world_position, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    let rect = spot_shadow.atlas_rect[shadow_index];
    let scale = rect.z;
    let uv = (ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5)) * scale + rect.xy;
    let depth = ndc.z + rect.w;
    let slot_min = rect.xy;
    let slot_max = rect.xy + scale;
    let in_bounds = uv.x >= slot_min.x && uv.x <= slot_max.x && uv.y >= slot_min.y && uv.y <= slot_max.y && ndc.z >= 0.0 && ndc.z <= 1.0;
    if !in_bounds {
        return 1.0;
    }
    let texel = 1.0 / 2048.0;
    let safe_min = slot_min + scale * 0.01;
    let safe_max = slot_max - scale * 0.01;
    var blocker_sum = 0.0;
    var blockers = 0u;
    for (var index = 0; index < 8; index = index + 1) {
        let sample_uv = clamp(uv + POISSON_8[index] * texel * 15.0, safe_min, safe_max);
        let sampled = textureSampleLevel(spotlight_atlas, shadow_sampler, sample_uv, 0);
        if sampled > depth {
            blocker_sum += sampled;
            blockers += 1u;
        }
    }
    if blockers == 0u {
        return 1.0;
    }
    let average = blocker_sum / f32(blockers);
    let penumbra = (average - depth) / max(1.0 - average, 0.001);
    let filter_radius = clamp(penumbra * 30.0, 0.5, 6.0);
    var visibility = 0.0;
    for (var index = 0; index < 16; index = index + 1) {
        let sample_uv = clamp(uv + POISSON_16[index] * texel * filter_radius, safe_min, safe_max);
        let sampled = textureSampleLevel(spotlight_atlas, shadow_sampler, sample_uv, 0);
        visibility += select(0.0, 1.0, depth >= sampled);
    }
    return visibility / 16.0;
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(7) normal: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) emissive: vec4<f32>,
    @location(8) albedo_metallic: vec4<f32>,
    @location(9) layers: vec4<f32>,
    @location(13) uv: vec2<f32>,
}
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(2) @interpolate(flat) emissive: vec3<f32>,
    @location(3) view_normal: vec3<f32>,
    @location(4) world_position: vec3<f32>,
    @location(5) world_normal: vec3<f32>,
    @location(6) @interpolate(flat) albedo: vec3<f32>,
    @location(7) @interpolate(flat) material: vec2<f32>,
    @location(8) @interpolate(flat) base_layer: u32,
    @location(9) uv: vec2<f32>,
    @location(10) @interpolate(flat) normal_layer: u32,
    @location(11) @interpolate(flat) orm_layer: u32,
    @location(12) @interpolate(flat) emissive_layer: u32,
}
struct GeometryOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
}
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * d * d, 0.0001);
}
fn geometry_schlick(n_dot: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return n_dot / (n_dot * (1.0 - k) + k);
}
fn fresnel(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}
fn range_attenuation(range: f32, distance: f32) -> f32 {
    if range <= 0.0 {
        return 1.0;
    }
    let clamped = max(distance, 0.01);
    return max(min(1.0 - pow(distance / range, 4.0), 1.0), 0.0) / (clamped * clamped);
}
fn spot_attenuation(spot_direction: vec3<f32>, to_light: vec3<f32>, outer_cos: f32, inner_cos: f32) -> f32 {
    let actual = dot(normalize(spot_direction), normalize(-to_light));
    if actual > outer_cos {
        return select(smoothstep(outer_cos, inner_cos, actual), 1.0, actual >= inner_cos);
    }
    return 0.0;
}
fn brdf(
    normal: vec3<f32>,
    view: vec3<f32>,
    light: vec3<f32>,
    radiance: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    let half = normalize(view + light);
    let n_dot_l = max(dot(normal, light), 0.0);
    let n_dot_v = max(dot(normal, view), 0.0001);
    let n_dot_h = max(dot(normal, half), 0.0);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_schlick(n_dot_v, roughness) * geometry_schlick(n_dot_l, roughness);
    let f = fresnel(max(dot(half, view), 0.0), f0);
    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic) * albedo / 3.14159265;
    return (diffuse + specular) * radiance * n_dot_l;
}
fn lighting(
    albedo: vec3<f32>,
    world_position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    frag_coord: vec2<f32>,
) -> vec3<f32> {
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let n_dot_v = max(dot(normal, view), 0.0001);
    let f_roughness = f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - n_dot_v, 0.0, 1.0), 5.0);
    let irradiance = textureSampleLevel(irradiance_map, ibl_sampler, normal, 0.0).rgb;
    let prefiltered = textureSampleLevel(prefiltered_map, ibl_sampler, reflect(-view, normal), roughness * 4.0).rgb;
    let env_brdf = textureSampleLevel(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
    let diffuse_ibl = irradiance * albedo * (vec3<f32>(1.0) - f_roughness) * (1.0 - metallic);
    let specular_ibl = prefiltered * (f_roughness * env_brdf.x + env_brdf.y);
    var color = (diffuse_ibl + specular_ibl) * occlusion;
    let sun = lights.sun_direction.xyz;
    let view_depth = -(camera.view * vec4<f32>(world_position, 1.0)).z;
    let sun_shadow = calculate_shadow(world_position, normal, view_depth);
    color += brdf(normal, view, sun, lights.sun_color.rgb * sun_shadow, albedo, metallic, roughness, f0);
    let grid = light_grid[get_cluster_index(frag_coord, view_depth)];
    for (var slot = 0u; slot < grid.count; slot = slot + 1u) {
        let index = light_indices[grid.offset + slot];
        let to_light = lights.point_position[index].xyz - world_position;
        let distance = max(length(to_light), 0.0001);
        let attenuation = range_attenuation(lights.point_position[index].w, distance);
        let spot = spot_attenuation(
            lights.point_direction[index].xyz,
            to_light,
            lights.point_direction[index].w,
            lights.point_color[index].w,
        );
        let spot_visibility = spot_shadow_factor(world_position, i32(lights.point_shadow[index].x));
        let point_visibility = point_shadow_factor(index, world_position);
        let radiance =
            lights.point_color[index].rgb * attenuation * spot * spot_visibility * point_visibility;
        color += brdf(normal, view, to_light / distance, radiance, albedo, metallic, roughness, f0);
    }
    return color;
}
@vertex fn vs(in: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let world = model * vec4<f32>(in.position, 1.0);
    let clip = camera.view_projection * world;
    let world_normal = (model * vec4<f32>(in.normal, 0.0)).xyz;
    let view_normal = (camera.view * vec4<f32>(world_normal, 0.0)).xyz;
    let model_scale = vec3<f32>(length(model[0].xyz), length(model[1].xyz), length(model[2].xyz));
    return VertexOutput(
        clip,
        vec4<f32>(in.color, 1.0),
        in.emissive.rgb,
        view_normal,
        world.xyz,
        world_normal,
        in.albedo_metallic.rgb,
        vec2<f32>(in.albedo_metallic.w, in.emissive.w),
        u32(in.layers.x),
        select(planar_uv(in.position * model_scale, in.normal), in.uv, in.layers.x >= 4.0),
        u32(in.layers.y),
        u32(in.layers.z),
        u32(in.layers.w),
    );
}
fn planar_uv(local_position: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let axis = abs(normal);
    var uv = local_position.xy;
    if axis.x >= axis.y && axis.x >= axis.z {
        uv = local_position.zy;
    } else if axis.y >= axis.z {
        uv = local_position.xz;
    }
    return uv * 0.5;
}
@fragment fn fs(in: VertexOutput) -> GeometryOutput {
    let geometric_normal = normalize(in.world_normal);
    let tangent_normal = textureSample(normal_textures, albedo_sampler, in.uv, in.normal_layer).xyz * 2.0 - 1.0;
    let normal = normalize(cotangent_frame(geometric_normal, in.world_position, in.uv) * tangent_normal);
    let view = normalize(camera.camera_position.xyz - in.world_position);
    let albedo = in.color.rgb * in.albedo * textureSample(albedo_textures, albedo_sampler, in.uv, in.base_layer).rgb;
    let orm = textureSample(orm_textures, albedo_sampler, in.uv, in.orm_layer).rgb;
    let metallic = in.material.x * orm.b;
    let roughness = clamp(max(in.material.y, 0.04) * orm.g, 0.04, 1.0);
    let lit = lighting(albedo, in.world_position, normal, view, metallic, roughness, orm.r, in.position.xy);
    let emissive_color = in.emissive * textureSample(emissive_textures, albedo_sampler, in.uv, in.emissive_layer).rgb;
    let shaded = lit + emissive_color;
    var out: GeometryOutput;
    out.color = vec4<f32>(shaded, 1.0);
    out.normal = vec4<f32>(normalize(in.view_normal), 1.0);
    return out;
}
