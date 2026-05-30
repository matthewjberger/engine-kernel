use nalgebra_glm::{Mat4, Quat, Vec2, Vec3};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use web_time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Entity {
    index: u32,
    generation: u32,
}

#[derive(Clone, Copy)]
struct Location {
    table: usize,
    row: usize,
}

#[derive(Clone, Copy)]
struct LocalTransform {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
}

impl Default for LocalTransform {
    fn default() -> Self {
        Self {
            translation: Vec3::zeros(),
            rotation: Quat::identity(),
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

fn local_transform_matrix(transform: &LocalTransform) -> Mat4 {
    nalgebra_glm::translation(&transform.translation)
        * nalgebra_glm::quat_to_mat4(&transform.rotation.normalize())
        * nalgebra_glm::scaling(&transform.scale)
}

#[derive(Clone, Copy)]
struct GlobalTransform(Mat4);

impl Default for GlobalTransform {
    fn default() -> Self {
        Self(Mat4::identity())
    }
}

fn transform_translation(matrix: &Mat4) -> Vec3 {
    nalgebra_glm::vec3(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)])
}

fn transform_up(matrix: &Mat4) -> Vec3 {
    nalgebra_glm::vec3(matrix[(0, 1)], matrix[(1, 1)], matrix[(2, 1)])
}

fn transform_forward(matrix: &Mat4) -> Vec3 {
    nalgebra_glm::vec3(-matrix[(0, 2)], -matrix[(1, 2)], -matrix[(2, 2)])
}

#[derive(Clone, Copy, Default)]
struct Parent(Option<Entity>);

#[derive(Clone, Copy, Default)]
struct LocalTransformDirty;

#[derive(Clone, Copy, Default)]
struct RenderMesh(u32);

#[derive(Clone, Copy, Default)]
struct Emissive(Vec3);

#[derive(Clone, Copy, Default, PartialEq)]
enum LightKind {
    #[default]
    Directional,
    Point,
    Spot,
}

#[derive(Clone, Copy, Default)]
struct Light {
    color: Vec3,
    intensity: f32,
    kind: LightKind,
    range: f32,
    inner_cone: f32,
    outer_cone: f32,
}

#[derive(Clone, Copy)]
struct Material {
    albedo: Vec3,
    metallic: f32,
    roughness: f32,
    base_layer: u32,
    normal_layer: u32,
    orm_layer: u32,
    emissive_layer: u32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            metallic: 0.0,
            roughness: 0.6,
            base_layer: 0,
            normal_layer: 0,
            orm_layer: 0,
            emissive_layer: 0,
        }
    }
}

#[derive(Clone, Default)]
struct Skin {
    joints: Vec<Entity>,
    inverse_bind: Vec<Mat4>,
}

#[derive(Clone, Copy, Default)]
struct SkinnedMesh(u32);

#[derive(Clone, Copy, PartialEq)]
enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Copy, PartialEq)]
enum Interpolation {
    Linear,
    Step,
}

#[derive(Clone)]
enum SamplerOutput {
    Vec3(Vec<Vec3>),
    Quat(Vec<Quat>),
}

#[derive(Clone)]
struct AnimationSampler {
    input: Vec<f32>,
    output: SamplerOutput,
    interpolation: Interpolation,
}

#[derive(Clone)]
struct AnimationChannel {
    target_node: usize,
    property: AnimationProperty,
    sampler: AnimationSampler,
}

#[derive(Clone)]
struct AnimationClip {
    duration: f32,
    channels: Vec<AnimationChannel>,
}

#[derive(Clone)]
struct AnimationPlayer {
    clips: Vec<AnimationClip>,
    current_clip: Option<usize>,
    time: f32,
    speed: f32,
    looping: bool,
    playing: bool,
    node_index_to_entity: HashMap<usize, Entity>,
}

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self {
            clips: Vec::new(),
            current_clip: None,
            time: 0.0,
            speed: 1.0,
            looping: true,
            playing: true,
            node_index_to_entity: HashMap::new(),
        }
    }
}

enum AnimationValue {
    Vec3(Vec3),
    Quat(Quat),
}

fn interpolate_vec3(start: &Vec3, end: &Vec3, factor: f32, interpolation: Interpolation) -> Vec3 {
    match interpolation {
        Interpolation::Step => *start,
        Interpolation::Linear => start + (end - start) * factor,
    }
}

fn interpolate_quat(start: &Quat, end: &Quat, factor: f32, interpolation: Interpolation) -> Quat {
    let start_normalized = start.normalize();
    match interpolation {
        Interpolation::Step => start_normalized,
        Interpolation::Linear => {
            let end_normalized = end.normalize();
            let aligned = if start_normalized.dot(&end_normalized) < 0.0 {
                -end_normalized
            } else {
                end_normalized
            };
            nalgebra_glm::quat_slerp(&start_normalized, &aligned, factor).normalize()
        }
    }
}

fn sample_animation_channel(channel: &AnimationChannel, time: f32) -> Option<AnimationValue> {
    let sampler = &channel.sampler;
    let times = &sampler.input;
    if times.is_empty() {
        return None;
    }
    let last = times.len() - 1;
    let (index_a, index_b, factor) = if time <= times[0] {
        (0, 0, 0.0)
    } else if time >= times[last] {
        (last, last, 0.0)
    } else {
        let next = times.partition_point(|&keyframe_time| keyframe_time <= time);
        let key = next - 1;
        let span = times[next] - times[key];
        let factor = if span > 0.0 {
            (time - times[key]) / span
        } else {
            0.0
        };
        (key, next, factor)
    };
    match &sampler.output {
        SamplerOutput::Vec3(values) => {
            let start = values.get(index_a)?;
            let end = values.get(index_b)?;
            Some(AnimationValue::Vec3(interpolate_vec3(
                start,
                end,
                factor,
                sampler.interpolation,
            )))
        }
        SamplerOutput::Quat(values) => {
            let start = values.get(index_a)?;
            let end = values.get(index_b)?;
            Some(AnimationValue::Quat(interpolate_quat(
                start,
                end,
                factor,
                sampler.interpolation,
            )))
        }
    }
}

#[derive(Clone, Copy)]
struct PerspectiveCamera {
    y_fov_radians: f32,
    z_near: f32,
    z_far: Option<f32>,
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            y_fov_radians: 45.0_f32.to_radians(),
            z_near: 0.01,
            z_far: None,
        }
    }
}

fn perspective_matrix(camera: &PerspectiveCamera, aspect_ratio: f32) -> Mat4 {
    let focal = 1.0 / (camera.y_fov_radians / 2.0).tan();
    let z_near = camera.z_near;
    let (depth_scale, depth_bias) = match camera.z_far {
        Some(z_far) => (z_near / (z_far - z_near), z_near * z_far / (z_far - z_near)),
        None => (0.0, z_near),
    };
    Mat4::new(
        focal / aspect_ratio,
        0.0,
        0.0,
        0.0,
        0.0,
        focal,
        0.0,
        0.0,
        0.0,
        0.0,
        depth_scale,
        depth_bias,
        0.0,
        0.0,
        -1.0,
        0.0,
    )
}

#[derive(Clone, Copy, Default)]
struct Camera {
    projection: PerspectiveCamera,
}

#[derive(Clone, Copy)]
struct PanOrbitCamera {
    focus: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
    target_focus: Vec3,
    target_radius: f32,
    target_yaw: f32,
    target_pitch: f32,
    orbit_sensitivity: f32,
    pan_sensitivity: f32,
    zoom_sensitivity: f32,
    orbit_smoothness: f32,
    pan_smoothness: f32,
    zoom_smoothness: f32,
    pitch_lower: f32,
    pitch_upper: f32,
    zoom_lower: f32,
    enabled: bool,
}

impl Default for PanOrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::zeros(),
            radius: 10.0,
            yaw: 0.0,
            pitch: 0.0,
            target_focus: Vec3::zeros(),
            target_radius: 10.0,
            target_yaw: 0.0,
            target_pitch: 0.0,
            orbit_sensitivity: 1.0,
            pan_sensitivity: 1.0,
            zoom_sensitivity: 1.0,
            orbit_smoothness: 0.1,
            pan_smoothness: 0.02,
            zoom_smoothness: 0.1,
            pitch_lower: -(std::f32::consts::FRAC_PI_2 - 0.01),
            pitch_upper: std::f32::consts::FRAC_PI_2 - 0.01,
            zoom_lower: 0.05,
            enabled: true,
        }
    }
}

fn pan_orbit_position_rotation(focus: Vec3, yaw: f32, pitch: f32, radius: f32) -> (Vec3, Quat) {
    let yaw_quat = nalgebra_glm::quat_angle_axis(yaw, &Vec3::y());
    let pitch_quat = nalgebra_glm::quat_angle_axis(-pitch, &Vec3::x());
    let rotation = yaw_quat * pitch_quat;
    let position = focus + nalgebra_glm::quat_rotate_vec3(&rotation, &Vec3::new(0.0, 0.0, radius));
    (position, rotation)
}

macro_rules! ecs {
    (@bits $index:expr,) => {};
    (@bits $index:expr, $constant:ident $($rest:ident)*) => {
        const $constant: u64 = 1u64 << $index;
        ecs!(@bits $index + 1, $($rest)*);
    };
    ($($constant:ident => $name:ty, $field:ident);* $(;)?) => {
        ecs!(@bits 0u64, $($constant)*);

        const COMPONENT_COUNT: usize = [$($constant),*].len();

        #[derive(Default)]
        struct Table {
            mask:     u64,
            entities: Vec<Entity>,
            $($field:  Vec<($name, u32)>,)*
            add_single:    [Option<usize>; COMPONENT_COUNT],
            remove_single: [Option<usize>; COMPONENT_COUNT],
            add_multi:     HashMap<u64, usize>,
            remove_multi:  HashMap<u64, usize>,
        }

        fn table_push(table: &mut Table, entity: Entity, tick: u32) -> usize {
            let row = table.entities.len();
            table.entities.push(entity);
            $(if table.mask & $constant != 0 { table.$field.push((Default::default(), tick)); })*
            row
        }

        fn table_swap_remove(table: &mut Table, row: usize) -> Option<Entity> {
            let last = table.entities.len().saturating_sub(1);
            let moved = if row < last { Some(table.entities[last]) } else { None };
            table.entities.swap_remove(row);
            $(if table.mask & $constant != 0 { table.$field.swap_remove(row); })*
            moved
        }

        fn table_move_row(source: &mut Table, row: usize, destination: &mut Table, tick: u32) -> (usize, Option<Entity>) {
            destination.entities.push(source.entities[row]);
            $(if destination.mask & $constant != 0 {
                destination.$field.push(
                    if source.mask & $constant != 0 { std::mem::take(&mut source.$field[row]) }
                    else { (Default::default(), tick) }
                );
            })*
            let destination_row = destination.entities.len() - 1;
            (destination_row, table_swap_remove(source, row))
        }
    }
}

ecs!(
    LOCAL_TRANSFORM       => LocalTransform,      local_transforms;
    GLOBAL_TRANSFORM      => GlobalTransform,     global_transforms;
    PARENT                => Parent,              parents;
    LOCAL_TRANSFORM_DIRTY => LocalTransformDirty, local_transform_dirty;
    CAMERA                => Camera,              cameras;
    PAN_ORBIT_CAMERA      => PanOrbitCamera,      pan_orbit_cameras;
    RENDER_MESH           => RenderMesh,          render_meshes;
    EMISSIVE              => Emissive,            emissives;
    LIGHT                 => Light,               lights;
    MATERIAL              => Material,            materials;
    SKIN                  => Skin,                skins;
    SKINNED_MESH          => SkinnedMesh,         skinned_meshes;
    ANIMATION_PLAYER      => AnimationPlayer,     animation_players;
);

type System = fn(&mut World);

#[derive(Default)]
struct Schedule {
    systems: Vec<(&'static str, System)>,
}

fn schedule_push(schedule: &mut Schedule, name: &'static str, system: System) {
    schedule.systems.push((name, system));
}

fn schedule_insert_before(
    schedule: &mut Schedule,
    target: &str,
    name: &'static str,
    system: System,
) {
    let position = schedule
        .systems
        .iter()
        .position(|(existing, _)| *existing == target)
        .unwrap_or(schedule.systems.len());
    schedule.systems.insert(position, (name, system));
}

fn schedule_insert_after(
    schedule: &mut Schedule,
    target: &str,
    name: &'static str,
    system: System,
) {
    let position = schedule
        .systems
        .iter()
        .position(|(existing, _)| *existing == target)
        .map_or(schedule.systems.len(), |index| index + 1);
    schedule.systems.insert(position, (name, system));
}

fn schedule_run(schedule: &Schedule, world: &mut World) {
    for (_, system) in &schedule.systems {
        system(world);
    }
}

#[derive(Default)]
struct Timing {
    last_frame: Option<Instant>,
}

#[derive(Default)]
struct TransformState {
    dirty_entities: Vec<Entity>,
    children_cache: HashMap<Entity, Vec<Entity>>,
    children_cache_valid: bool,
}

#[derive(Default)]
struct Input {
    left_pressed: bool,
    right_pressed: bool,
    cursor: Vec2,
    cursor_initialized: bool,
    position_delta: Vec2,
    wheel_delta: f32,
}

enum InputEvent {
    Cursor(Vec2),
    Button(MouseButton, bool),
    Wheel(f32),
}

#[derive(Clone, Copy)]
struct SpawnCommand {
    mask: u64,
    transform: LocalTransform,
    render_mesh: u32,
    emissive: Vec3,
    material: Material,
}

#[derive(Default)]
struct Resources {
    delta_time: f32,
    viewport: (f32, f32),
    active_camera: Option<Entity>,
    schedule: Schedule,
    timing: Timing,
    transform_state: TransformState,
    input: Input,
    events: Vec<InputEvent>,
    commands: Vec<SpawnCommand>,
    bloom_enabled: bool,
    ssr_enabled: bool,
}

#[derive(Default)]
pub struct World {
    tables: Vec<Table>,
    table_map: HashMap<u64, usize>,
    query_cache: HashMap<u64, Vec<usize>>,
    locations: Vec<Option<Location>>,
    generations: Vec<u32>,
    free_ids: Vec<u32>,
    current_tick: u32,
    resources: Resources,
}

fn new_world() -> World {
    World {
        current_tick: 1,
        resources: Resources {
            viewport: (1.0, 1.0),
            bloom_enabled: true,
            ssr_enabled: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn alloc_entity(world: &mut World) -> Entity {
    if let Some(index) = world.free_ids.pop() {
        Entity {
            index,
            generation: world.generations[index as usize],
        }
    } else {
        let index = world.generations.len() as u32;
        world.generations.push(0);
        Entity {
            index,
            generation: 0,
        }
    }
}

fn is_live(world: &World, entity: Entity) -> bool {
    world.generations.get(entity.index as usize).copied() == Some(entity.generation)
}

fn get_location(world: &World, entity: Entity) -> Option<Location> {
    world
        .locations
        .get(entity.index as usize)
        .and_then(|location| *location)
}

fn set_location(world: &mut World, entity: Entity, location: Location) {
    let index = entity.index as usize;
    if index >= world.locations.len() {
        world.locations.resize(index + 1, None);
    }
    world.locations[index] = Some(location);
}

fn component_index(bit: u64) -> usize {
    bit.trailing_zeros() as usize
}

fn get_or_create_table(world: &mut World, mask: u64) -> usize {
    if let Some(&index) = world.table_map.get(&mask) {
        return index;
    }
    let index = world.tables.len();
    world.tables.push(Table {
        mask,
        ..Default::default()
    });
    world.table_map.insert(mask, index);
    for (required, indices) in world.query_cache.iter_mut() {
        if mask & required == *required {
            indices.push(index);
        }
    }
    index
}

fn query_tables(world: &mut World, required: u64) -> &[usize] {
    let tables = &world.tables;
    world.query_cache.entry(required).or_insert_with(|| {
        tables
            .iter()
            .enumerate()
            .filter(|(_, table)| table.mask & required == required)
            .map(|(index, _)| index)
            .collect()
    })
}

fn migrate(world: &mut World, entity: Entity, location: Location, new_mask: u64) {
    let source = location.table;
    let old_mask = world.tables[source].mask;
    let delta = old_mask ^ new_mask;
    let adding = (new_mask & delta) != 0;
    let tick = world.current_tick;

    let destination = if delta.count_ones() == 1 {
        let component = component_index(delta);
        let cached = if adding {
            world.tables[source].add_single[component]
        } else {
            world.tables[source].remove_single[component]
        };
        cached.unwrap_or_else(|| {
            let destination = get_or_create_table(world, new_mask);
            if adding {
                world.tables[source].add_single[component] = Some(destination);
            } else {
                world.tables[source].remove_single[component] = Some(destination);
            }
            destination
        })
    } else {
        let cached = if adding {
            world.tables[source].add_multi.get(&delta).copied()
        } else {
            world.tables[source].remove_multi.get(&delta).copied()
        };
        cached.unwrap_or_else(|| {
            let destination = get_or_create_table(world, new_mask);
            if adding {
                world.tables[source].add_multi.insert(delta, destination);
            } else {
                world.tables[source].remove_multi.insert(delta, destination);
            }
            destination
        })
    };

    let (source_table, destination_table) = if source < destination {
        let (left, right) = world.tables.split_at_mut(destination);
        (&mut left[source], &mut right[0])
    } else {
        let (left, right) = world.tables.split_at_mut(source);
        (&mut right[0], &mut left[destination])
    };
    let (destination_row, back_filled) =
        table_move_row(source_table, location.row, destination_table, tick);
    set_location(
        world,
        entity,
        Location {
            table: destination,
            row: destination_row,
        },
    );
    if let Some(moved) = back_filled {
        set_location(
            world,
            moved,
            Location {
                table: source,
                row: location.row,
            },
        );
    }
}

fn spawn(world: &mut World, mask: u64) -> Entity {
    let entity = alloc_entity(world);
    let table = get_or_create_table(world, mask);
    let tick = world.current_tick;
    let row = table_push(&mut world.tables[table], entity, tick);
    set_location(world, entity, Location { table, row });
    entity
}

fn remove_components(world: &mut World, entity: Entity, removed: u64) {
    if !is_live(world, entity) {
        return;
    }
    let location = get_location(world, entity).unwrap();
    let old_mask = world.tables[location.table].mask;
    if old_mask & removed == 0 {
        return;
    }
    migrate(world, entity, location, old_mask & !removed);
}

fn entity_has(world: &World, entity: Entity, mask: u64) -> bool {
    is_live(world, entity)
        && get_location(world, entity)
            .is_some_and(|location| world.tables[location.table].mask & mask == mask)
}

fn component<T>(
    world: &World,
    entity: Entity,
    bit: u64,
    read: impl FnOnce(&Table, usize) -> T,
) -> Option<T> {
    let location = get_location(world, entity)?;
    let table = &world.tables[location.table];
    (table.mask & bit != 0).then(|| read(table, location.row))
}

fn component_mut<'world, T>(
    world: &'world mut World,
    entity: Entity,
    bit: u64,
    read: impl FnOnce(&'world mut Table, usize, u32) -> &'world mut T,
) -> Option<&'world mut T> {
    let location = get_location(world, entity)?;
    let tick = world.current_tick;
    let table = &mut world.tables[location.table];
    if table.mask & bit != 0 {
        Some(read(table, location.row, tick))
    } else {
        None
    }
}

fn get_local_transform(world: &World, entity: Entity) -> Option<LocalTransform> {
    component(world, entity, LOCAL_TRANSFORM, |table, row| {
        table.local_transforms[row].0
    })
}

fn get_local_transform_mut(world: &mut World, entity: Entity) -> Option<&mut LocalTransform> {
    component_mut(world, entity, LOCAL_TRANSFORM, |table, row, tick| {
        table.local_transforms[row].1 = tick;
        &mut table.local_transforms[row].0
    })
}

fn get_global_transform(world: &World, entity: Entity) -> Option<GlobalTransform> {
    component(world, entity, GLOBAL_TRANSFORM, |table, row| {
        table.global_transforms[row].0
    })
}

fn set_global_transform(world: &mut World, entity: Entity, value: GlobalTransform) {
    let Some(location) = get_location(world, entity) else {
        return;
    };
    let tick = world.current_tick;
    let table = &mut world.tables[location.table];
    if table.mask & GLOBAL_TRANSFORM != 0 {
        table.global_transforms[location.row] = (value, tick);
    }
}

fn get_parent(world: &World, entity: Entity) -> Option<Parent> {
    component(world, entity, PARENT, |table, row| table.parents[row].0)
}

fn get_camera(world: &World, entity: Entity) -> Option<Camera> {
    component(world, entity, CAMERA, |table, row| table.cameras[row].0)
}

fn get_pan_orbit_camera_mut(world: &mut World, entity: Entity) -> Option<&mut PanOrbitCamera> {
    component_mut(world, entity, PAN_ORBIT_CAMERA, |table, row, tick| {
        table.pan_orbit_cameras[row].1 = tick;
        &mut table.pan_orbit_cameras[row].0
    })
}

fn get_emissive_mut(world: &mut World, entity: Entity) -> Option<&mut Emissive> {
    component_mut(world, entity, EMISSIVE, |table, row, tick| {
        table.emissives[row].1 = tick;
        &mut table.emissives[row].0
    })
}

fn get_render_mesh_mut(world: &mut World, entity: Entity) -> Option<&mut RenderMesh> {
    component_mut(world, entity, RENDER_MESH, |table, row, tick| {
        table.render_meshes[row].1 = tick;
        &mut table.render_meshes[row].0
    })
}

fn get_skinned_mesh_mut(world: &mut World, entity: Entity) -> Option<&mut SkinnedMesh> {
    component_mut(world, entity, SKINNED_MESH, |table, row, tick| {
        table.skinned_meshes[row].1 = tick;
        &mut table.skinned_meshes[row].0
    })
}

fn get_skin_mut(world: &mut World, entity: Entity) -> Option<&mut Skin> {
    component_mut(world, entity, SKIN, |table, row, tick| {
        table.skins[row].1 = tick;
        &mut table.skins[row].0
    })
}

fn get_animation_player_mut(world: &mut World, entity: Entity) -> Option<&mut AnimationPlayer> {
    component_mut(world, entity, ANIMATION_PLAYER, |table, row, tick| {
        table.animation_players[row].1 = tick;
        &mut table.animation_players[row].0
    })
}

fn get_light_mut(world: &mut World, entity: Entity) -> Option<&mut Light> {
    component_mut(world, entity, LIGHT, |table, row, tick| {
        table.lights[row].1 = tick;
        &mut table.lights[row].0
    })
}

fn get_material_mut(world: &mut World, entity: Entity) -> Option<&mut Material> {
    component_mut(world, entity, MATERIAL, |table, row, tick| {
        table.materials[row].1 = tick;
        &mut table.materials[row].0
    })
}

fn collect_entities(world: &mut World, mask: u64) -> Vec<Entity> {
    let mut entities = Vec::new();
    for table in query_tables(world, mask).to_vec() {
        entities.extend_from_slice(&world.tables[table].entities);
    }
    entities
}

fn mark_local_transform_dirty(world: &mut World, entity: Entity) {
    if !world.resources.transform_state.children_cache_valid {
        rebuild_children_cache(world);
    }

    let is_parent = world
        .resources
        .transform_state
        .children_cache
        .contains_key(&entity);
    if !entity_has(world, entity, PARENT) && !is_parent {
        let global = match get_local_transform(world, entity) {
            Some(local) => local_transform_matrix(&local),
            None => Mat4::identity(),
        };
        set_global_transform(world, entity, GlobalTransform(global));
        return;
    }

    let mut stack = vec![entity];
    while let Some(current) = stack.pop() {
        world.resources.transform_state.dirty_entities.push(current);
        if let Some(children) = world.resources.transform_state.children_cache.get(&current) {
            stack.extend(children.iter().copied());
        }
    }
}

fn rebuild_children_cache(world: &mut World) {
    world.resources.transform_state.children_cache.clear();

    let mut relationships = Vec::new();
    for table in query_tables(world, PARENT).to_vec() {
        let table = &world.tables[table];
        for (entity, parent) in table.entities.iter().zip(table.parents.iter()) {
            if let Some(parent_entity) = parent.0.0 {
                relationships.push((parent_entity, *entity));
            }
        }
    }

    for (parent, child) in relationships {
        world
            .resources
            .transform_state
            .children_cache
            .entry(parent)
            .or_default()
            .push(child);
    }
    for children in world.resources.transform_state.children_cache.values_mut() {
        children.sort_unstable_by_key(|entity| (entity.index, entity.generation));
    }
    world.resources.transform_state.children_cache_valid = true;
}

fn global_transform_with_cycle_detection(
    world: &World,
    entity: Entity,
    visited: &mut HashSet<Entity>,
) -> Mat4 {
    if !visited.insert(entity) {
        return Mat4::identity();
    }
    let Some(local) = get_local_transform(world, entity) else {
        return Mat4::identity();
    };
    match get_parent(world, entity).and_then(|parent| parent.0) {
        Some(parent_entity) => {
            global_transform_with_cycle_detection(world, parent_entity, visited)
                * local_transform_matrix(&local)
        }
        None => local_transform_matrix(&local),
    }
}

fn update_global_transforms_system(world: &mut World) {
    let mut dirty = std::mem::take(&mut world.resources.transform_state.dirty_entities);
    dirty.extend(collect_entities(
        world,
        LOCAL_TRANSFORM_DIRTY | LOCAL_TRANSFORM | GLOBAL_TRANSFORM,
    ));
    for &entity in &dirty {
        if entity_has(world, entity, LOCAL_TRANSFORM_DIRTY) {
            remove_components(world, entity, LOCAL_TRANSFORM_DIRTY);
        }
    }
    dirty.sort_unstable_by_key(|entity| (entity.index, entity.generation));
    dirty.dedup();

    let mut visited = HashSet::new();
    for entity in dirty {
        let global = if entity_has(world, entity, PARENT) {
            visited.clear();
            global_transform_with_cycle_detection(world, entity, &mut visited)
        } else {
            match get_local_transform(world, entity) {
                Some(local) => local_transform_matrix(&local),
                None => Mat4::identity(),
            }
        };
        set_global_transform(world, entity, GlobalTransform(global));
    }
}

fn update_animation_players_system(world: &mut World) {
    let delta_time = world.resources.delta_time;
    for table in &mut world.tables {
        if table.mask & ANIMATION_PLAYER == 0 {
            continue;
        }
        for slot in table.animation_players.iter_mut() {
            let player = &mut slot.0;
            if !player.playing {
                continue;
            }
            let Some(clip_index) = player.current_clip else {
                continue;
            };
            let Some(clip) = player.clips.get(clip_index) else {
                continue;
            };
            let duration = clip.duration;
            player.time += delta_time * player.speed;
            if player.time >= duration {
                if player.looping {
                    player.time = if duration > 0.0 {
                        player.time % duration
                    } else {
                        0.0
                    };
                } else {
                    player.time = duration;
                    player.playing = false;
                }
            }
        }
    }
}

fn apply_animations_system(world: &mut World) {
    let mut updates: Vec<(Entity, AnimationProperty, AnimationValue)> = Vec::new();
    for table in &world.tables {
        if table.mask & ANIMATION_PLAYER == 0 {
            continue;
        }
        for slot in table.animation_players.iter() {
            let player = &slot.0;
            let Some(clip_index) = player.current_clip else {
                continue;
            };
            let Some(clip) = player.clips.get(clip_index) else {
                continue;
            };
            for channel in &clip.channels {
                let Some(&entity) = player.node_index_to_entity.get(&channel.target_node) else {
                    continue;
                };
                if let Some(value) = sample_animation_channel(channel, player.time) {
                    updates.push((entity, channel.property, value));
                }
            }
        }
    }
    for (entity, property, value) in updates {
        if let Some(transform) = get_local_transform_mut(world, entity) {
            match (property, value) {
                (AnimationProperty::Translation, AnimationValue::Vec3(translation)) => {
                    transform.translation = translation;
                }
                (AnimationProperty::Scale, AnimationValue::Vec3(scale)) => {
                    transform.scale = scale;
                }
                (AnimationProperty::Rotation, AnimationValue::Quat(rotation)) => {
                    transform.rotation = rotation.normalize();
                }
                _ => {}
            }
        }
        mark_local_transform_dirty(world, entity);
    }
}

const SPIN_SPEED: f32 = 0.8;

fn spin_system(world: &mut World) {
    let delta_time = world.resources.delta_time;
    for entity in collect_entities(world, RENDER_MESH) {
        if entity_has(world, entity, EMISSIVE) {
            continue;
        }
        if let Some(local) = get_local_transform_mut(world, entity) {
            let spin = nalgebra_glm::quat_angle_axis(SPIN_SPEED * delta_time, &Vec3::y());
            local.rotation = spin * local.rotation;
        }
        mark_local_transform_dirty(world, entity);
    }
}

fn move_lights_system(world: &mut World) {
    let delta_time = world.resources.delta_time;
    for entity in collect_entities(world, LIGHT | RENDER_MESH | LOCAL_TRANSFORM) {
        if let Some(local) = get_local_transform_mut(world, entity) {
            let radius = (local.translation.x * local.translation.x
                + local.translation.z * local.translation.z)
                .sqrt();
            let angle = local.translation.z.atan2(local.translation.x) + delta_time * 0.4;
            local.translation.x = angle.cos() * radius;
            local.translation.z = angle.sin() * radius;
        }
        mark_local_transform_dirty(world, entity);
    }
}

fn input_system(world: &mut World) {
    let events = std::mem::take(&mut world.resources.events);
    let input = &mut world.resources.input;
    for event in events {
        match event {
            InputEvent::Cursor(cursor) => {
                if input.cursor_initialized {
                    input.position_delta += cursor - input.cursor;
                } else {
                    input.cursor_initialized = true;
                }
                input.cursor = cursor;
            }
            InputEvent::Button(MouseButton::Left, pressed) => input.left_pressed = pressed,
            InputEvent::Button(MouseButton::Right, pressed) => input.right_pressed = pressed,
            InputEvent::Button(_, _) => {}
            InputEvent::Wheel(delta) => input.wheel_delta += delta,
        }
    }
}

fn process_commands_system(world: &mut World) {
    let commands = std::mem::take(&mut world.resources.commands);
    for command in commands {
        let entity = spawn(world, command.mask);
        if let Some(local) = get_local_transform_mut(world, entity) {
            *local = command.transform;
        }
        if let Some(render_mesh) = get_render_mesh_mut(world, entity) {
            render_mesh.0 = command.render_mesh;
        }
        if let Some(emissive) = get_emissive_mut(world, entity) {
            emissive.0 = command.emissive;
        }
        if let Some(material) = get_material_mut(world, entity) {
            *material = command.material;
        }
        mark_local_transform_dirty(world, entity);
    }
}

fn camera_y_fov(world: &World, camera: Entity) -> f32 {
    get_camera(world, camera)
        .map(|camera| camera.projection.y_fov_radians)
        .unwrap_or(45.0_f32.to_radians())
}

fn lerp_and_snap(from: f32, to: f32, smoothness: f32, delta_time: f32) -> f32 {
    let factor = smoothness.powi(7);
    let result = from + (to - from) * (1.0 - factor.powf(delta_time));
    if smoothness < 1.0 && (result - to).abs() < 0.001 {
        to
    } else {
        result
    }
}

fn lerp_and_snap_vec3(from: Vec3, to: Vec3, smoothness: f32, delta_time: f32) -> Vec3 {
    let factor = smoothness.powi(7);
    let result = from + (to - from) * (1.0 - factor.powf(delta_time));
    if smoothness < 1.0 && (result - to).magnitude() < 0.001 {
        to
    } else {
        result
    }
}

fn pan_orbit_camera_system(world: &mut World) {
    let Some(camera) = world.resources.active_camera else {
        return;
    };
    if !entity_has(world, camera, PAN_ORBIT_CAMERA) {
        return;
    }

    let viewport = Vec2::new(
        world.resources.viewport.0.max(1.0),
        world.resources.viewport.1.max(1.0),
    );
    let y_fov = camera_y_fov(world, camera);
    let orbit = world.resources.input.left_pressed;
    let pan = world.resources.input.right_pressed;
    let position_delta = world.resources.input.position_delta;
    let wheel_delta = world.resources.input.wheel_delta;

    let Some(pan_orbit) = get_pan_orbit_camera_mut(world, camera) else {
        return;
    };
    if !pan_orbit.enabled {
        return;
    }

    if orbit {
        let delta_yaw = (position_delta.x / viewport.x) * std::f32::consts::TAU;
        let delta_pitch = (position_delta.y / viewport.y) * std::f32::consts::PI;
        pan_orbit.target_yaw -= delta_yaw * pan_orbit.orbit_sensitivity;
        pan_orbit.target_pitch += delta_pitch * pan_orbit.orbit_sensitivity;
        pan_orbit.target_pitch = pan_orbit
            .target_pitch
            .clamp(pan_orbit.pitch_lower, pan_orbit.pitch_upper);
    }

    if pan {
        let (_, rotation) = pan_orbit_position_rotation(
            pan_orbit.target_focus,
            pan_orbit.target_yaw,
            pan_orbit.target_pitch,
            pan_orbit.target_radius,
        );
        let right = nalgebra_glm::quat_rotate_vec3(&rotation, &Vec3::x());
        let up = nalgebra_glm::quat_rotate_vec3(&rotation, &Vec3::y());
        let world_per_pixel = (pan_orbit.target_radius * 2.0 * (y_fov * 0.5).tan()) / viewport.y;
        let scale = world_per_pixel * pan_orbit.pan_sensitivity;
        pan_orbit.target_focus += right * -position_delta.x * scale + up * position_delta.y * scale;
    }

    if wheel_delta != 0.0 {
        let zoom = -wheel_delta * pan_orbit.target_radius * 0.2 * pan_orbit.zoom_sensitivity;
        pan_orbit.target_radius = (pan_orbit.target_radius + zoom).max(pan_orbit.zoom_lower);
    }

    let delta_time = world.resources.delta_time;
    let Some(pan_orbit) = get_pan_orbit_camera_mut(world, camera) else {
        return;
    };
    pan_orbit.yaw = lerp_and_snap(
        pan_orbit.yaw,
        pan_orbit.target_yaw,
        pan_orbit.orbit_smoothness,
        delta_time,
    );
    pan_orbit.pitch = lerp_and_snap(
        pan_orbit.pitch,
        pan_orbit.target_pitch,
        pan_orbit.orbit_smoothness,
        delta_time,
    );
    pan_orbit.radius = lerp_and_snap(
        pan_orbit.radius,
        pan_orbit.target_radius,
        pan_orbit.zoom_smoothness,
        delta_time,
    );
    pan_orbit.focus = lerp_and_snap_vec3(
        pan_orbit.focus,
        pan_orbit.target_focus,
        pan_orbit.pan_smoothness,
        delta_time,
    );

    let (position, rotation) = pan_orbit_position_rotation(
        pan_orbit.focus,
        pan_orbit.yaw,
        pan_orbit.pitch,
        pan_orbit.radius,
    );

    if let Some(local) = get_local_transform_mut(world, camera) {
        local.translation = position;
        local.rotation = rotation;
    }
    mark_local_transform_dirty(world, camera);
}

fn camera_view(world: &World) -> Option<Mat4> {
    let camera_entity = world.resources.active_camera?;
    let global = get_global_transform(world, camera_entity)?.0;
    let position = transform_translation(&global);
    let target = position + transform_forward(&global);
    let up = transform_up(&global);
    Some(nalgebra_glm::look_at(&position, &target, &up))
}

fn camera_projection(world: &World, aspect_ratio: f32) -> Option<Mat4> {
    let camera_entity = world.resources.active_camera?;
    let camera = get_camera(world, camera_entity)?;
    Some(perspective_matrix(&camera.projection, aspect_ratio))
}

fn reverse_z_ortho_light(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    let width = right - left;
    let height = top - bottom;
    let depth = far - near;
    Mat4::new(
        2.0 / width,
        0.0,
        0.0,
        -(right + left) / width,
        0.0,
        2.0 / height,
        0.0,
        -(top + bottom) / height,
        0.0,
        0.0,
        1.0 / depth,
        -near / depth,
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

fn reverse_z_perspective(y_fov: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let focal = 1.0 / (y_fov / 2.0).tan();
    let depth = far - near;
    Mat4::new(
        focal / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        focal,
        0.0,
        0.0,
        0.0,
        0.0,
        near / depth,
        near * far / depth,
        0.0,
        0.0,
        -1.0,
        0.0,
    )
}

fn frustum_corners_world(view: &Mat4, y_fov: f32, aspect: f32, near: f32, far: f32) -> [Vec3; 8] {
    let inverse_view = nalgebra_glm::inverse(view);
    let tan_half = (y_fov / 2.0).tan();
    let near_height = near * tan_half;
    let near_width = near_height * aspect;
    let far_height = far * tan_half;
    let far_width = far_height * aspect;
    let view_corners = [
        nalgebra_glm::vec3(-near_width, -near_height, -near),
        nalgebra_glm::vec3(near_width, -near_height, -near),
        nalgebra_glm::vec3(near_width, near_height, -near),
        nalgebra_glm::vec3(-near_width, near_height, -near),
        nalgebra_glm::vec3(-far_width, -far_height, -far),
        nalgebra_glm::vec3(far_width, -far_height, -far),
        nalgebra_glm::vec3(far_width, far_height, -far),
        nalgebra_glm::vec3(-far_width, far_height, -far),
    ];
    let mut corners = [Vec3::zeros(); 8];
    for (index, corner) in view_corners.iter().enumerate() {
        let world = inverse_view * nalgebra_glm::vec4(corner.x, corner.y, corner.z, 1.0);
        corners[index] = nalgebra_glm::vec3(world.x, world.y, world.z);
    }
    corners
}

fn cascade_view_projection(
    corners: &[Vec3; 8],
    light_direction: Vec3,
    resolution: f32,
) -> (Mat4, f32) {
    let mut center = Vec3::zeros();
    for corner in corners {
        center += corner;
    }
    center /= 8.0;
    let mut max_radius = 0.0f32;
    for corner in corners {
        max_radius = max_radius.max(nalgebra_glm::length(&(corner - center)));
    }
    let up = if light_direction.y.abs() > 0.99 {
        Vec3::x()
    } else {
        Vec3::y()
    };
    let direction = light_direction.normalize();
    let view = nalgebra_glm::look_at(&(center - direction * max_radius * 4.0), &center, &up);
    let mut min = nalgebra_glm::vec3(f32::MAX, f32::MAX, f32::MAX);
    let mut max = nalgebra_glm::vec3(f32::MIN, f32::MIN, f32::MIN);
    for corner in corners {
        let light_space = view * nalgebra_glm::vec4(corner.x, corner.y, corner.z, 1.0);
        min = nalgebra_glm::vec3(
            min.x.min(light_space.x),
            min.y.min(light_space.y),
            min.z.min(light_space.z),
        );
        max = nalgebra_glm::vec3(
            max.x.max(light_space.x),
            max.y.max(light_space.y),
            max.z.max(light_space.z),
        );
    }
    let padding = (max.x - min.x).max(max.y - min.y) * 0.1;
    min.x -= padding;
    max.x += padding;
    min.y -= padding;
    max.y += padding;
    let z_multiplier = 10.0;
    let min_z = if min.z < 0.0 {
        min.z * z_multiplier
    } else {
        min.z / z_multiplier
    };
    let max_z = if max.z < 0.0 {
        max.z / z_multiplier
    } else {
        max.z * z_multiplier
    };
    let projection = reverse_z_ortho_light(min.x, max.x, min.y, max.y, min_z, max_z);
    (projection * view, (max.x - min.x) / resolution)
}

fn cascade_splits(near: f32) -> [f32; 4] {
    let scale = (near / 0.01).max(1e-6);
    [10.0 * scale, 40.0 * scale, 150.0 * scale, 500.0 * scale]
}

fn camera_near_fov(world: &World) -> (f32, f32) {
    if let Some(entity) = world.resources.active_camera
        && let Some(camera) = get_camera(world, entity)
    {
        (camera.projection.z_near, camera.projection.y_fov_radians)
    } else {
        (0.01, 45.0_f32.to_radians())
    }
}

fn view_projection_matrix(world: &World, aspect_ratio: f32) -> Option<Mat4> {
    Some(camera_projection(world, aspect_ratio)? * camera_view(world)?)
}

fn mesh_instances(world: &World, handle: u32) -> Vec<(Mat4, Vec3, Material)> {
    let mut instances = Vec::new();
    for table in &world.tables {
        if table.mask & (RENDER_MESH | GLOBAL_TRANSFORM) != (RENDER_MESH | GLOBAL_TRANSFORM) {
            continue;
        }
        let emissive_table = table.mask & EMISSIVE != 0;
        let material_table = table.mask & MATERIAL != 0;
        for row in 0..table.entities.len() {
            if table.render_meshes[row].0.0 != handle {
                continue;
            }
            let model = table.global_transforms[row].0.0;
            let emissive = if emissive_table {
                table.emissives[row].0.0
            } else {
                Vec3::zeros()
            };
            let material = if material_table {
                table.materials[row].0
            } else {
                Material::default()
            };
            instances.push((model, emissive, material));
        }
    }
    instances
}

fn skinned_instances(world: &World) -> Vec<(u32, Vec<Mat4>, Vec3, Material)> {
    let required = SKINNED_MESH | SKIN | GLOBAL_TRANSFORM;
    let mut instances = Vec::new();
    for table in &world.tables {
        if table.mask & required != required {
            continue;
        }
        let emissive_table = table.mask & EMISSIVE != 0;
        let material_table = table.mask & MATERIAL != 0;
        for row in 0..table.entities.len() {
            let handle = table.skinned_meshes[row].0.0;
            let skin = &table.skins[row].0;
            let matrices: Vec<Mat4> = skin
                .joints
                .iter()
                .enumerate()
                .map(|(index, &joint)| {
                    let global = get_global_transform(world, joint)
                        .map_or_else(Mat4::identity, |transform| transform.0);
                    let inverse_bind = skin
                        .inverse_bind
                        .get(index)
                        .copied()
                        .unwrap_or_else(Mat4::identity);
                    global * inverse_bind
                })
                .collect();
            let emissive = if emissive_table {
                table.emissives[row].0.0
            } else {
                Vec3::zeros()
            };
            let material = if material_table {
                table.materials[row].0
            } else {
                Material::default()
            };
            instances.push((handle, matrices, emissive, material));
        }
    }
    instances
}

fn gather_lights(world: &World) -> ([f32; 144], Vec<Mat4>) {
    let mut data = [0.0f32; 144];
    data[0] = 0.03;
    data[1] = 0.03;
    data[2] = 0.04;
    let mut point_count = 0usize;
    let mut point_shadow_count = 0usize;
    let mut spot_shadows = Vec::new();
    for table in &world.tables {
        if table.mask & (LIGHT | GLOBAL_TRANSFORM) != (LIGHT | GLOBAL_TRANSFORM) {
            continue;
        }
        for row in 0..table.entities.len() {
            let global = table.global_transforms[row].0.0;
            let light = table.lights[row].0;
            let color = light.color * light.intensity;
            match light.kind {
                LightKind::Directional => {
                    let direction = -transform_forward(&global);
                    data[4] = direction.x;
                    data[5] = direction.y;
                    data[6] = direction.z;
                    data[7] = 1.0;
                    data[8] = color.x;
                    data[9] = color.y;
                    data[10] = color.z;
                }
                LightKind::Point | LightKind::Spot if point_count < 8 => {
                    let position = transform_translation(&global);
                    let slot = 16 + point_count * 4;
                    data[slot] = position.x;
                    data[slot + 1] = position.y;
                    data[slot + 2] = position.z;
                    data[slot + 3] = light.range;
                    let color_slot = 48 + point_count * 4;
                    data[color_slot] = color.x;
                    data[color_slot + 1] = color.y;
                    data[color_slot + 2] = color.z;
                    let direction_slot = 80 + point_count * 4;
                    let shadow_slot = 112 + point_count * 4;
                    data[shadow_slot] = -1.0;
                    data[shadow_slot + 1] = -1.0;
                    if light.kind == LightKind::Spot {
                        let direction = transform_forward(&global);
                        data[direction_slot] = direction.x;
                        data[direction_slot + 1] = direction.y;
                        data[direction_slot + 2] = direction.z;
                        data[direction_slot + 3] = light.outer_cone.cos();
                        data[color_slot + 3] = light.inner_cone.cos();
                        if spot_shadows.len() < 4 {
                            let up = if direction.y.abs() > 0.99 {
                                Vec3::x()
                            } else {
                                Vec3::y()
                            };
                            let view =
                                nalgebra_glm::look_at(&position, &(position + direction), &up);
                            let projection = reverse_z_perspective(
                                light.outer_cone * 2.0,
                                1.0,
                                0.1,
                                light.range.max(1.0),
                            );
                            data[shadow_slot] = spot_shadows.len() as f32;
                            spot_shadows.push(projection * view);
                        }
                    } else {
                        data[direction_slot + 3] = -2.0;
                        data[color_slot + 3] = -2.0;
                        if point_shadow_count < 4 {
                            data[shadow_slot + 1] = point_shadow_count as f32;
                            point_shadow_count += 1;
                        }
                    }
                    point_count += 1;
                }
                _ => {}
            }
        }
    }
    data[12] = point_count as f32;
    (data, spot_shadows)
}

fn timing_system(world: &mut World) {
    let now = Instant::now();
    world.resources.delta_time = world
        .resources
        .timing
        .last_frame
        .map_or(0.0, |last_frame| (now - last_frame).as_secs_f32());
    world.resources.timing.last_frame = Some(now);
    world.current_tick = world.current_tick.wrapping_add(1);
}

fn run_frame_systems(world: &mut World) {
    let schedule = std::mem::take(&mut world.resources.schedule);
    schedule_run(&schedule, world);
    world.resources.schedule = schedule;
}

pub trait State {
    fn initialize(&mut self, world: &mut World);
    fn run_systems(&mut self, _world: &mut World) {}
}

#[derive(Default)]
pub struct Demo;

impl State for Demo {
    fn initialize(&mut self, world: &mut World) {
        schedule_push(
            &mut world.resources.schedule,
            "transforms",
            update_global_transforms_system,
        );
        schedule_insert_before(
            &mut world.resources.schedule,
            "transforms",
            "spin",
            spin_system,
        );
        schedule_insert_before(
            &mut world.resources.schedule,
            "transforms",
            "move_lights",
            move_lights_system,
        );
        schedule_insert_before(
            &mut world.resources.schedule,
            "transforms",
            "animate",
            update_animation_players_system,
        );
        schedule_insert_before(
            &mut world.resources.schedule,
            "transforms",
            "apply_animations",
            apply_animations_system,
        );
        schedule_insert_after(
            &mut world.resources.schedule,
            "spin",
            "pan_orbit",
            pan_orbit_camera_system,
        );
        schedule_insert_before(&mut world.resources.schedule, "spin", "input", input_system);
        schedule_insert_before(
            &mut world.resources.schedule,
            "input",
            "commands",
            process_commands_system,
        );

        let camera = spawn(
            world,
            LOCAL_TRANSFORM | GLOBAL_TRANSFORM | CAMERA | PAN_ORBIT_CAMERA,
        );
        if let Some(pan_orbit) = get_pan_orbit_camera_mut(world, camera) {
            pan_orbit.yaw = 0.6;
            pan_orbit.target_yaw = 0.6;
            pan_orbit.pitch = 0.35;
            pan_orbit.target_pitch = 0.35;
        }
        world.resources.active_camera = Some(camera);
        pan_orbit_camera_system(world);

        let sun = spawn(world, LOCAL_TRANSFORM | GLOBAL_TRANSFORM | LIGHT);
        if let Some(local) = get_local_transform_mut(world, sun) {
            local.rotation = nalgebra_glm::quat_angle_axis(-1.0, &Vec3::x());
        }
        if let Some(light) = get_light_mut(world, sun) {
            light.color = nalgebra_glm::vec3(1.0, 0.96, 0.9);
            light.intensity = 2.5;
        }
        mark_local_transform_dirty(world, sun);

        let lamp = spawn(world, LOCAL_TRANSFORM | GLOBAL_TRANSFORM | LIGHT);
        if let Some(local) = get_local_transform_mut(world, lamp) {
            local.translation = nalgebra_glm::vec3(0.0, 0.5, 0.0);
        }
        if let Some(light) = get_light_mut(world, lamp) {
            light.color = nalgebra_glm::vec3(1.0, 0.55, 0.2);
            light.intensity = 4.0;
            light.kind = LightKind::Point;
            light.range = 7.0;
        }
        mark_local_transform_dirty(world, lamp);

        let spot = spawn(world, LOCAL_TRANSFORM | GLOBAL_TRANSFORM | LIGHT);
        if let Some(local) = get_local_transform_mut(world, spot) {
            local.translation = nalgebra_glm::vec3(0.0, 6.0, 0.0);
            local.rotation =
                nalgebra_glm::quat_angle_axis(-std::f32::consts::FRAC_PI_2, &Vec3::x());
        }
        if let Some(light) = get_light_mut(world, spot) {
            light.color = nalgebra_glm::vec3(0.4, 0.7, 1.0);
            light.intensity = 60.0;
            light.kind = LightKind::Spot;
            light.range = 16.0;
            light.inner_cone = 0.2;
            light.outer_cone = 0.35;
        }
        mark_local_transform_dirty(world, spot);

        let orbit_lights = [
            (
                nalgebra_glm::vec3(1.0, 0.25, 0.25),
                nalgebra_glm::vec3(3.0, 0.6, 0.0),
            ),
            (
                nalgebra_glm::vec3(0.3, 1.0, 0.4),
                nalgebra_glm::vec3(0.0, 0.6, 3.2),
            ),
            (
                nalgebra_glm::vec3(0.35, 0.5, 1.0),
                nalgebra_glm::vec3(-3.4, 0.6, 1.6),
            ),
            (
                nalgebra_glm::vec3(1.0, 0.85, 0.35),
                nalgebra_glm::vec3(1.8, 0.6, -3.0),
            ),
        ];
        for (color, position) in orbit_lights {
            let orb = spawn(
                world,
                LOCAL_TRANSFORM | GLOBAL_TRANSFORM | RENDER_MESH | EMISSIVE | LIGHT,
            );
            if let Some(local) = get_local_transform_mut(world, orb) {
                local.translation = position;
                local.scale = nalgebra_glm::vec3(0.18, 0.18, 0.18);
            }
            if let Some(render_mesh) = get_render_mesh_mut(world, orb) {
                render_mesh.0 = 1;
            }
            if let Some(emissive) = get_emissive_mut(world, orb) {
                emissive.0 = color * 8.0;
            }
            if let Some(light) = get_light_mut(world, orb) {
                light.color = color;
                light.intensity = 40.0;
                light.kind = LightKind::Point;
                light.range = 12.0;
            }
            mark_local_transform_dirty(world, orb);
        }

        world.resources.commands.push(SpawnCommand {
            mask: LOCAL_TRANSFORM | GLOBAL_TRANSFORM | RENDER_MESH | EMISSIVE,
            transform: LocalTransform {
                translation: nalgebra_glm::vec3(0.0, 3.0, 0.0),
                scale: nalgebra_glm::vec3(0.4, 0.4, 0.4),
                ..Default::default()
            },
            render_mesh: 0,
            emissive: nalgebra_glm::vec3(4.0, 2.2, 0.8),
            material: Material::default(),
        });

        world.resources.commands.push(SpawnCommand {
            mask: LOCAL_TRANSFORM | GLOBAL_TRANSFORM | RENDER_MESH | MATERIAL | EMISSIVE,
            transform: LocalTransform {
                translation: nalgebra_glm::vec3(0.0, -1.0, 0.0),
                scale: nalgebra_glm::vec3(10.0, 0.1, 10.0),
                ..Default::default()
            },
            render_mesh: 0,
            emissive: Vec3::zeros(),
            material: Material {
                albedo: nalgebra_glm::vec3(0.7, 0.7, 0.7),
                metallic: 0.0,
                roughness: 0.25,
                base_layer: 2,
                normal_layer: 2,
                orm_layer: 0,
                emissive_layer: 0,
            },
        });

        let grid = 50i32;
        for x in -grid / 2..grid / 2 {
            for z in -grid / 2..grid / 2 {
                world.resources.commands.push(SpawnCommand {
                    mask: LOCAL_TRANSFORM | GLOBAL_TRANSFORM | RENDER_MESH | MATERIAL | EMISSIVE,
                    transform: LocalTransform {
                        translation: nalgebra_glm::vec3(x as f32 * 1.6, 0.5, z as f32 * 1.6),
                        scale: nalgebra_glm::vec3(0.3, 0.3, 0.3),
                        ..Default::default()
                    },
                    render_mesh: 0,
                    emissive: Vec3::zeros(),
                    material: Material {
                        albedo: nalgebra_glm::vec3(0.8, 0.45, 0.2),
                        metallic: 0.1,
                        roughness: 0.5,
                        base_layer: 1,
                        normal_layer: 1,
                        orm_layer: 0,
                        emissive_layer: 0,
                    },
                });
            }
        }
    }
}

const CUBE_VERTICES: [f32; 324] = [
    1., -1., -1., 1., 1., 1., 1., 0., 0., 1., 1., -1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1.,
    1., 1., 0., 0., 1., -1., -1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1.,
    -1., 1., 1., 1., 1., 1., 0., 0., -1., -1., -1., 0.55, 0.55, 0.55, -1., 0., 0., -1., 1., 1.,
    0.55, 0.55, 0.55, -1., 0., 0., -1., 1., -1., 0.55, 0.55, 0.55, -1., 0., 0., -1., -1., -1.,
    0.55, 0.55, 0.55, -1., 0., 0., -1., -1., 1., 0.55, 0.55, 0.55, -1., 0., 0., -1., 1., 1., 0.55,
    0.55, 0.55, -1., 0., 0., -1., 1., -1., 0.9, 0.9, 0.9, 0., 1., 0., -1., 1., 1., 0.9, 0.9, 0.9,
    0., 1., 0., 1., 1., 1., 0.9, 0.9, 0.9, 0., 1., 0., -1., 1., -1., 0.9, 0.9, 0.9, 0., 1., 0., 1.,
    1., 1., 0.9, 0.9, 0.9, 0., 1., 0., 1., 1., -1., 0.9, 0.9, 0.9, 0., 1., 0., -1., -1., -1., 0.45,
    0.45, 0.45, 0., -1., 0., 1., -1., 1., 0.45, 0.45, 0.45, 0., -1., 0., -1., -1., 1., 0.45, 0.45,
    0.45, 0., -1., 0., -1., -1., -1., 0.45, 0.45, 0.45, 0., -1., 0., 1., -1., -1., 0.45, 0.45,
    0.45, 0., -1., 0., 1., -1., 1., 0.45, 0.45, 0.45, 0., -1., 0., -1., -1., 1., 0.8, 0.8, 0.8, 0.,
    0., 1., 1., -1., 1., 0.8, 0.8, 0.8, 0., 0., 1., 1., 1., 1., 0.8, 0.8, 0.8, 0., 0., 1., -1.,
    -1., 1., 0.8, 0.8, 0.8, 0., 0., 1., 1., 1., 1., 0.8, 0.8, 0.8, 0., 0., 1., -1., 1., 1., 0.8,
    0.8, 0.8, 0., 0., 1., -1., -1., -1., 0.65, 0.65, 0.65, 0., 0., -1., 1., 1., -1., 0.65, 0.65,
    0.65, 0., 0., -1., 1., -1., -1., 0.65, 0.65, 0.65, 0., 0., -1., -1., -1., -1., 0.65, 0.65,
    0.65, 0., 0., -1., -1., 1., -1., 0.65, 0.65, 0.65, 0., 0., -1., 1., 1., -1., 0.65, 0.65, 0.65,
    0., 0., -1.,
];

fn sphere_mesh(rings: usize, sectors: usize) -> Vec<f32> {
    let mut vertices = Vec::with_capacity(rings * sectors * 6 * 11);
    let point = |ring: usize, sector: usize| {
        let theta = ring as f32 / rings as f32 * std::f32::consts::PI;
        let phi = sector as f32 / sectors as f32 * std::f32::consts::TAU;
        (
            [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ],
            [sector as f32 / sectors as f32, ring as f32 / rings as f32],
        )
    };
    for ring in 0..rings {
        for sector in 0..sectors {
            let (p00, uv00) = point(ring, sector);
            let (p10, uv10) = point(ring + 1, sector);
            let (p01, uv01) = point(ring, sector + 1);
            let (p11, uv11) = point(ring + 1, sector + 1);
            for (position, uv) in [
                (p00, uv00),
                (p10, uv10),
                (p11, uv11),
                (p00, uv00),
                (p11, uv11),
                (p01, uv01),
            ] {
                vertices.extend_from_slice(&[
                    position[0],
                    position[1],
                    position[2],
                    1.0,
                    1.0,
                    1.0,
                    position[0],
                    position[1],
                    position[2],
                    uv[0],
                    uv[1],
                ]);
            }
        }
    }
    vertices
}

const CLUSTER_X: u32 = 16;
const CLUSTER_Y: u32 = 9;
const CLUSTER_Z: u32 = 24;
const CLUSTER_TOTAL: u32 = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
const MAX_LIGHTS_PER_CLUSTER: u32 = 8;

const CLUSTER_BOUNDS_SHADER: &str = include_str!("shaders/cluster_bounds.wgsl");

const CLUSTER_ASSIGN_SHADER: &str = include_str!("shaders/cluster_assign.wgsl");

const IBL_COMMON: &str = include_str!("shaders/ibl_common.wgsl");

const EQUIRECT_TO_CUBE_SHADER: &str = include_str!("shaders/equirect_to_cube.wgsl");

const CUBEMAP_MIPGEN_SHADER: &str = include_str!("shaders/cubemap_mipgen.wgsl");

const FILTER_ENVMAP_SHADER: &str = include_str!("shaders/filter_envmap.wgsl");

const BRDF_LUT_SHADER: &str = include_str!("shaders/brdf_lut.wgsl");

const SKIN_COMPUTE_SHADER: &str = include_str!("shaders/skin_compute.wgsl");

const MESH_CULL_SHADER: &str = include_str!("shaders/mesh_cull.wgsl");

const SHADER: &str = include_str!("shaders/mesh.wgsl");

const SKY_SHADER: &str = include_str!("shaders/sky.wgsl");

const SHADOW_SHADER: &str = include_str!("shaders/shadow.wgsl");

const POINT_SHADOW_SHADER: &str = include_str!("shaders/point_shadow.wgsl");

const CUBE_FACES: [(Vec3, Vec3); 6] = [
    (
        nalgebra_glm::Vec3::new(1.0, 0.0, 0.0),
        nalgebra_glm::Vec3::new(0.0, -1.0, 0.0),
    ),
    (
        nalgebra_glm::Vec3::new(-1.0, 0.0, 0.0),
        nalgebra_glm::Vec3::new(0.0, -1.0, 0.0),
    ),
    (
        nalgebra_glm::Vec3::new(0.0, 1.0, 0.0),
        nalgebra_glm::Vec3::new(0.0, 0.0, 1.0),
    ),
    (
        nalgebra_glm::Vec3::new(0.0, -1.0, 0.0),
        nalgebra_glm::Vec3::new(0.0, 0.0, -1.0),
    ),
    (
        nalgebra_glm::Vec3::new(0.0, 0.0, 1.0),
        nalgebra_glm::Vec3::new(0.0, -1.0, 0.0),
    ),
    (
        nalgebra_glm::Vec3::new(0.0, 0.0, -1.0),
        nalgebra_glm::Vec3::new(0.0, -1.0, 0.0),
    ),
];

const FULLSCREEN_VERTEX: &str = include_str!("shaders/fullscreen.wgsl");

const BRIGHT_SHADER: &str = include_str!("shaders/bright.wgsl");

const BLUR_SHADER: &str = include_str!("shaders/blur.wgsl");

const SSAO_SHADER: &str = include_str!("shaders/ssao.wgsl");

const SSAO_BLUR_SHADER: &str = include_str!("shaders/ssao_blur.wgsl");

const COMPOSITE_SHADER: &str = include_str!("shaders/composite.wgsl");

const FXAA_SHADER: &str = include_str!("shaders/fxaa.wgsl");

const AUTO_EXPOSURE_SHADER: &str = include_str!("shaders/auto_exposure.wgsl");

const SSR_SHADER: &str = include_str!("shaders/ssr.wgsl");

const SSR_BLUR_SHADER: &str = include_str!("shaders/ssr_blur.wgsl");

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone, Copy)]
enum Clear {
    Color(wgpu::Color),
    Depth(f32),
}

struct ResourceDesc {
    external: bool,
    format: wgpu::TextureFormat,
    clear: Clear,
}

struct GraphPass {
    node: Box<dyn PassNode>,
    bindings: HashMap<&'static str, usize>,
}

#[derive(Default)]
struct RenderGraph {
    resources: Vec<ResourceDesc>,
    passes: Vec<GraphPass>,
    execution_order: Vec<usize>,
    clears: HashSet<(usize, usize)>,
    stores: HashSet<(usize, usize)>,
    resource_physical: Vec<Option<usize>>,
    physical_formats: Vec<wgpu::TextureFormat>,
    physical_views: Vec<wgpu::TextureView>,
    size: (u32, u32),
}

struct PassContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    encoder: &'a mut wgpu::CommandEncoder,
    world: &'a World,
    aspect_ratio: f32,
    size: (u32, u32),
    resources: &'a [ResourceDesc],
    views: &'a [Option<&'a wgpu::TextureView>],
    bindings: &'a HashMap<&'static str, usize>,
    clears: &'a HashSet<(usize, usize)>,
    stores: &'a HashSet<(usize, usize)>,
    pass_index: usize,
}

trait PassNode {
    fn reads(&self) -> Vec<&'static str> {
        Vec::new()
    }
    fn color_writes(&self) -> Vec<&'static str> {
        Vec::new()
    }
    fn depth_write(&self) -> Option<&'static str> {
        None
    }
    fn execute(&mut self, context: &mut PassContext);
}

fn write_slots(node: &dyn PassNode) -> Vec<&'static str> {
    let mut slots = node.color_writes();
    slots.extend(node.depth_write());
    slots
}

fn add_color_resource(
    graph: &mut RenderGraph,
    external: bool,
    format: wgpu::TextureFormat,
    clear_color: wgpu::Color,
) -> usize {
    graph.resources.push(ResourceDesc {
        external,
        format,
        clear: Clear::Color(clear_color),
    });
    graph.resources.len() - 1
}

fn add_depth_resource(
    graph: &mut RenderGraph,
    format: wgpu::TextureFormat,
    clear_depth: f32,
) -> usize {
    graph.resources.push(ResourceDesc {
        external: false,
        format,
        clear: Clear::Depth(clear_depth),
    });
    graph.resources.len() - 1
}

fn add_pass(graph: &mut RenderGraph, node: Box<dyn PassNode>, bindings: &[(&'static str, usize)]) {
    graph.passes.push(GraphPass {
        node,
        bindings: bindings.iter().copied().collect(),
    });
}

fn binding_resources(pass: &GraphPass, slots: &[&'static str]) -> Vec<usize> {
    slots
        .iter()
        .filter_map(|slot| pass.bindings.get(slot).copied())
        .collect()
}

fn render_graph_compile(graph: &mut RenderGraph) {
    let pass_count = graph.passes.len();
    let resource_count = graph.resources.len();
    let reads: Vec<Vec<usize>> = graph
        .passes
        .iter()
        .map(|pass| binding_resources(pass, &pass.node.reads()))
        .collect();
    let writes: Vec<Vec<usize>> = graph
        .passes
        .iter()
        .map(|pass| binding_resources(pass, &write_slots(pass.node.as_ref())))
        .collect();

    let mut edges = Vec::new();
    let mut last_writer: HashMap<usize, usize> = HashMap::new();
    for index in 0..pass_count {
        for resource in reads[index].iter().chain(writes[index].iter()) {
            if let Some(&writer) = last_writer.get(resource)
                && writer != index
            {
                edges.push((writer, index));
            }
        }
        for resource in &writes[index] {
            last_writer.insert(*resource, index);
        }
    }

    let mut incoming = vec![Vec::new(); pass_count];
    for &(from, to) in &edges {
        incoming[to].push(from);
    }
    let mut keep = vec![false; pass_count];
    let mut frontier: Vec<usize> = (0..pass_count)
        .filter(|&index| {
            writes[index]
                .iter()
                .any(|&resource| graph.resources[resource].external)
        })
        .collect();
    for &seed in &frontier {
        keep[seed] = true;
    }
    while let Some(node) = frontier.pop() {
        for &producer in &incoming[node] {
            if !keep[producer] {
                keep[producer] = true;
                frontier.push(producer);
            }
        }
    }

    let mut adjacency = vec![Vec::new(); pass_count];
    let mut indegree = vec![0usize; pass_count];
    for &(from, to) in &edges {
        if keep[from] && keep[to] {
            adjacency[from].push(to);
            indegree[to] += 1;
        }
    }
    let mut ready: Vec<usize> = (0..pass_count)
        .filter(|&index| keep[index] && indegree[index] == 0)
        .collect();
    let mut order = Vec::with_capacity(pass_count);
    let mut cursor = 0;
    while cursor < ready.len() {
        let node = ready[cursor];
        cursor += 1;
        order.push(node);
        for &next in &adjacency[node] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push(next);
            }
        }
    }

    let mut cleared = HashSet::new();
    graph.clears.clear();
    for &index in &order {
        for &resource in &writes[index] {
            if cleared.insert(resource) {
                graph.clears.insert((index, resource));
            }
        }
    }

    graph.stores.clear();
    for resource in 0..resource_count {
        let mut producer: Option<usize> = None;
        for &index in &order {
            if reads[index].contains(&resource)
                && let Some(writer) = producer
            {
                graph.stores.insert((writer, resource));
            }
            if writes[index].contains(&resource) {
                producer = Some(index);
            }
        }
        if graph.resources[resource].external
            && let Some(writer) = producer
        {
            graph.stores.insert((writer, resource));
        }
    }

    let mut first_use = vec![usize::MAX; resource_count];
    let mut last_use = vec![0usize; resource_count];
    let mut used = vec![false; resource_count];
    for (position, &index) in order.iter().enumerate() {
        for &resource in reads[index].iter().chain(writes[index].iter()) {
            used[resource] = true;
            first_use[resource] = first_use[resource].min(position);
            last_use[resource] = last_use[resource].max(position);
        }
    }

    let mut transient: Vec<usize> = (0..resource_count)
        .filter(|&resource| used[resource] && !graph.resources[resource].external)
        .collect();
    transient.sort_by_key(|&resource| first_use[resource]);

    let mut resource_physical = vec![None; resource_count];
    let mut physical_formats: Vec<wgpu::TextureFormat> = Vec::new();
    let mut physical_last_use: Vec<usize> = Vec::new();
    for resource in transient {
        let format = graph.resources[resource].format;
        let reused = (0..physical_formats.len()).find(|&slot| {
            physical_formats[slot] == format && physical_last_use[slot] < first_use[resource]
        });
        let slot = match reused {
            Some(slot) => {
                physical_last_use[slot] = last_use[resource];
                slot
            }
            None => {
                physical_formats.push(format);
                physical_last_use.push(last_use[resource]);
                physical_formats.len() - 1
            }
        };
        resource_physical[resource] = Some(slot);
    }

    graph.resource_physical = resource_physical;
    graph.physical_formats = physical_formats;
    graph.physical_views.clear();
    graph.size = (0, 0);
    graph.execution_order = order;
}

fn ensure_resources(graph: &mut RenderGraph, device: &wgpu::Device, size: (u32, u32)) {
    if graph.size == size && graph.physical_views.len() == graph.physical_formats.len() {
        return;
    }
    graph.physical_views = graph
        .physical_formats
        .iter()
        .map(|&format| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: size.0.max(1),
                    height: size.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            texture.create_view(&Default::default())
        })
        .collect();
    graph.size = size;
}

fn store_op(context: &PassContext, resource: usize) -> wgpu::StoreOp {
    if context.stores.contains(&(context.pass_index, resource)) {
        wgpu::StoreOp::Store
    } else {
        wgpu::StoreOp::Discard
    }
}

fn read_view<'a>(context: &PassContext<'a>, slot: &str) -> &'a wgpu::TextureView {
    let resource = context.bindings[slot];
    context.views[resource].expect("unbound read resource")
}

fn color_attachment<'a>(
    context: &PassContext<'a>,
    slot: &str,
) -> (
    &'a wgpu::TextureView,
    wgpu::LoadOp<wgpu::Color>,
    wgpu::StoreOp,
) {
    let resource = context.bindings[slot];
    let view = context.views[resource].expect("unbound color attachment");
    let load = match (
        context.clears.contains(&(context.pass_index, resource)),
        context.resources[resource].clear,
    ) {
        (true, Clear::Color(color)) => wgpu::LoadOp::Clear(color),
        _ => wgpu::LoadOp::Load,
    };
    (view, load, store_op(context, resource))
}

fn depth_attachment<'a>(
    context: &PassContext<'a>,
    slot: &str,
) -> (&'a wgpu::TextureView, wgpu::LoadOp<f32>, wgpu::StoreOp) {
    let resource = context.bindings[slot];
    let view = context.views[resource].expect("unbound depth attachment");
    let load = match (
        context.clears.contains(&(context.pass_index, resource)),
        context.resources[resource].clear,
    ) {
        (true, Clear::Depth(depth)) => wgpu::LoadOp::Clear(depth),
        _ => wgpu::LoadOp::Load,
    };
    (view, load, store_op(context, resource))
}

fn fullscreen_pass(
    context: &mut PassContext,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    output: &str,
) {
    let (view, load, store) = color_attachment(context, output);
    let mut pass = context
        .encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load, store },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

fn render_graph_execute(
    graph: &mut RenderGraph,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    world: &World,
    aspect_ratio: f32,
    external: &wgpu::TextureView,
    size: (u32, u32),
) -> wgpu::CommandBuffer {
    ensure_resources(graph, device, size);

    let views: Vec<Option<&wgpu::TextureView>> = graph
        .resources
        .iter()
        .enumerate()
        .map(|(id, resource)| {
            if resource.external {
                Some(external)
            } else {
                graph.resource_physical[id].map(|slot| &graph.physical_views[slot])
            }
        })
        .collect();

    let mut encoder = device.create_command_encoder(&Default::default());
    for index in graph.execution_order.clone() {
        let bindings = graph.passes[index].bindings.clone();
        let mut context = PassContext {
            device,
            queue,
            encoder: &mut encoder,
            world,
            aspect_ratio,
            size,
            resources: &graph.resources,
            views: &views,
            bindings: &bindings,
            clears: &graph.clears,
            stores: &graph.stores,
            pass_index: index,
        };
        graph.passes[index].node.execute(&mut context);
    }
    encoder.finish()
}

struct MeshGpu {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    visible_indices_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    bounds_buffer: wgpu::Buffer,
    cull_buffer: wgpu::Buffer,
    objects_bind_group: Option<wgpu::BindGroup>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    point_bind_group: Option<wgpu::BindGroup>,
    cached_instances: Vec<f32>,
}

struct GeometryPass {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_view: wgpu::TextureView,
    cascade_buffers: [wgpu::Buffer; 4],
    cascade_bind_groups: [wgpu::BindGroup; 4],
    shadow_buffer: wgpu::Buffer,
    spot_atlas_view: wgpu::TextureView,
    spot_buffers: [wgpu::Buffer; 4],
    spot_bind_groups: [wgpu::BindGroup; 4],
    spot_shadow_buffer: wgpu::Buffer,
    cluster_uniform_buffer: wgpu::Buffer,
    cluster_lights_buffer: wgpu::Buffer,
    cluster_bounds_pipeline: wgpu::ComputePipeline,
    cluster_bounds_bind_group: wgpu::BindGroup,
    cluster_assign_pipeline: wgpu::ComputePipeline,
    cluster_assign_bind_group: wgpu::BindGroup,
    sky_pipeline: wgpu::RenderPipeline,
    sky_bind_group: wgpu::BindGroup,
    point_pipeline: wgpu::RenderPipeline,
    point_face_views: [wgpu::TextureView; 24],
    point_depth_view: wgpu::TextureView,
    point_face_buffers: [wgpu::Buffer; 24],
    point_face_bind_groups: [wgpu::BindGroup; 24],
    meshes: Vec<MeshGpu>,
    skinned: Vec<SkinnedGpu>,
    skin_pipeline: wgpu::ComputePipeline,
    joint_buffer: wgpu::Buffer,
    joint_capacity: u32,
    cull_pipeline: wgpu::ComputePipeline,
    camera_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

fn upload_instances(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut wgpu::Buffer,
    capacity: &mut u32,
    data: &[f32],
    count: u32,
) {
    if count > *capacity {
        *capacity = count.next_power_of_two();
        *buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: *capacity as u64 * 176,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
    if count > 0 {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }
}

fn upload_instances_diff(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut wgpu::Buffer,
    capacity: &mut u32,
    cached: &mut Vec<f32>,
    data: &[f32],
    count: u32,
) {
    if count > *capacity {
        *capacity = count.next_power_of_two();
        *buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: *capacity as u64 * 176,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        cached.clear();
    }
    if count == 0 {
        cached.clear();
        return;
    }
    if cached.len() != data.len() {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        cached.clear();
        cached.extend_from_slice(data);
        return;
    }
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut changed = 0usize;
    let mut open: Option<(usize, usize)> = None;
    for (record, (new, old)) in data
        .chunks_exact(44)
        .zip(cached.chunks_exact(44))
        .enumerate()
    {
        if new != old {
            changed += 1;
            open = Some(open.map_or((record, record + 1), |(start, _)| (start, record + 1)));
        } else if let Some(range) = open.take() {
            ranges.push(range);
        }
    }
    if let Some(range) = open.take() {
        ranges.push(range);
    }
    if ranges.is_empty() {
        return;
    }
    if changed as f32 / count as f32 > 0.3 {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    } else {
        for (start, end) in ranges {
            queue.write_buffer(
                buffer,
                start as u64 * 176,
                bytemuck::cast_slice(&data[start * 44..end * 44]),
            );
        }
    }
    cached.clear();
    cached.extend_from_slice(data);
}

struct GltfNode {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
    parent: Option<usize>,
    mesh: Option<usize>,
    skin: Option<usize>,
}

struct GltfSkin {
    joints: Vec<usize>,
    inverse_bind: Vec<Mat4>,
}

struct GltfImport {
    primitives: Vec<Vec<f32>>,
    skinned_primitives: Vec<Vec<u32>>,
    base_color: Option<Vec<u8>>,
    normal: Option<Vec<u8>>,
    orm: Option<Vec<u8>>,
    emissive: Option<Vec<u8>>,
    base_factor: Vec3,
    metallic: f32,
    roughness: f32,
    emissive_factor: Vec3,
    nodes: Vec<GltfNode>,
    skins: Vec<GltfSkin>,
    animations: Vec<AnimationClip>,
}

struct GltfScene {
    nodes: Vec<GltfNode>,
    material: Material,
    emissive: Vec3,
    mesh_base: u32,
    skinned_mesh_base: u32,
    skins: Vec<GltfSkin>,
    animations: Vec<AnimationClip>,
}

#[cfg(target_arch = "wasm32")]
fn fetch_gltf(_url: &str) -> Option<GltfImport> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_texture(buffers: &[gltf::buffer::Data], texture: Option<gltf::Image>) -> Option<Vec<u8>> {
    let bytes = match texture?.source() {
        gltf::image::Source::View { view, .. } => {
            let buffer = &buffers[view.buffer().index()].0;
            buffer[view.offset()..view.offset() + view.length()].to_vec()
        }
        gltf::image::Source::Uri { .. } => return None,
    };
    let decoded = image::load_from_memory(&bytes).ok()?.to_rgba8();
    Some(
        image::imageops::resize(&decoded, 512, 512, image::imageops::FilterType::Triangle)
            .into_raw(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_gltf(url: &str) -> Option<GltfImport> {
    use std::io::Read;
    let name = url.rsplit('/').next().unwrap_or("model.glb");
    let cache = std::env::temp_dir().join(format!("engine_codegolfed_{name}"));
    let bytes = match std::fs::read(&cache) {
        Ok(bytes) => bytes,
        Err(_) => {
            let mut bytes = Vec::new();
            ureq::get(url)
                .call()
                .ok()?
                .into_reader()
                .read_to_end(&mut bytes)
                .ok()?;
            let _ = std::fs::write(&cache, &bytes);
            bytes
        }
    };
    let (document, buffers, _images) = gltf::import_slice(&bytes).ok()?;
    let mut primitives: Vec<Vec<f32>> = Vec::new();
    let mut skinned_primitives: Vec<Vec<u32>> = Vec::new();
    let mut mesh_first = std::collections::HashMap::new();
    let mut skinned_mesh_first = std::collections::HashMap::new();
    let mut material = None;
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader =
                primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| &data.0[..]));
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            let joints = reader
                .read_joints(0)
                .map(|iter| iter.into_u16().collect::<Vec<[u16; 4]>>());
            let weights = reader
                .read_weights(0)
                .map(|iter| iter.into_f32().collect::<Vec<[f32; 4]>>());
            if let (Some(joints), Some(weights)) = (joints, weights) {
                skinned_mesh_first
                    .entry(mesh.index())
                    .or_insert(skinned_primitives.len());
                let mut vertices = Vec::with_capacity(indices.len() * 16);
                for &index in &indices {
                    let p = positions[index as usize];
                    let n = normals
                        .get(index as usize)
                        .copied()
                        .unwrap_or([0.0, 1.0, 0.0]);
                    let uv = uvs.get(index as usize).copied().unwrap_or([0.0, 0.0]);
                    let j = joints.get(index as usize).copied().unwrap_or([0, 0, 0, 0]);
                    let w = weights
                        .get(index as usize)
                        .copied()
                        .unwrap_or([1.0, 0.0, 0.0, 0.0]);
                    vertices.extend_from_slice(&[
                        p[0].to_bits(),
                        p[1].to_bits(),
                        p[2].to_bits(),
                        n[0].to_bits(),
                        n[1].to_bits(),
                        n[2].to_bits(),
                        uv[0].to_bits(),
                        uv[1].to_bits(),
                        w[0].to_bits(),
                        w[1].to_bits(),
                        w[2].to_bits(),
                        w[3].to_bits(),
                        j[0] as u32,
                        j[1] as u32,
                        j[2] as u32,
                        j[3] as u32,
                    ]);
                }
                skinned_primitives.push(vertices);
            } else {
                mesh_first.entry(mesh.index()).or_insert(primitives.len());
                let mut vertices = Vec::with_capacity(indices.len() * 11);
                for &index in &indices {
                    let p = positions[index as usize];
                    let n = normals
                        .get(index as usize)
                        .copied()
                        .unwrap_or([0.0, 1.0, 0.0]);
                    let uv = uvs.get(index as usize).copied().unwrap_or([0.0, 0.0]);
                    vertices.extend_from_slice(&[
                        p[0], p[1], p[2], 1.0, 1.0, 1.0, n[0], n[1], n[2], uv[0], uv[1],
                    ]);
                }
                primitives.push(vertices);
            }
            if material.is_none() {
                material = Some(primitive.material());
            }
        }
    }
    let material = material?;
    let pbr = material.pbr_metallic_roughness();
    let base_color = decode_texture(
        &buffers,
        pbr.base_color_texture().map(|i| i.texture().source()),
    );
    let normal = decode_texture(
        &buffers,
        material.normal_texture().map(|t| t.texture().source()),
    );
    let emissive = decode_texture(
        &buffers,
        material.emissive_texture().map(|i| i.texture().source()),
    );
    let metallic_roughness = decode_texture(
        &buffers,
        pbr.metallic_roughness_texture()
            .map(|i| i.texture().source()),
    );
    let occlusion = decode_texture(
        &buffers,
        material.occlusion_texture().map(|t| t.texture().source()),
    );
    let orm = (metallic_roughness.is_some() || occlusion.is_some()).then(|| {
        (0..512 * 512usize)
            .flat_map(|texel| {
                let r = occlusion.as_ref().map_or(255, |o| o[texel * 4]);
                let g = metallic_roughness
                    .as_ref()
                    .map_or(255, |m| m[texel * 4 + 1]);
                let b = metallic_roughness
                    .as_ref()
                    .map_or(255, |m| m[texel * 4 + 2]);
                [r, g, b, 255]
            })
            .collect()
    });
    let base_factor = pbr.base_color_factor();
    let emissive_factor = material.emissive_factor();

    let mut parents = vec![None; document.nodes().count()];
    for node in document.nodes() {
        for child in node.children() {
            parents[child.index()] = Some(node.index());
        }
    }
    let nodes = document
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            let mesh = node.mesh().and_then(|m| {
                if node.skin().is_some() {
                    skinned_mesh_first.get(&m.index()).copied()
                } else {
                    mesh_first.get(&m.index()).copied()
                }
            });
            GltfNode {
                translation: nalgebra_glm::vec3(translation[0], translation[1], translation[2]),
                rotation: Quat::new(rotation[3], rotation[0], rotation[1], rotation[2]),
                scale: nalgebra_glm::vec3(scale[0], scale[1], scale[2]),
                parent: parents[node.index()],
                mesh,
                skin: node.skin().map(|skin| skin.index()),
            }
        })
        .collect();

    let skins = document
        .skins()
        .map(|skin| {
            let reader = skin.reader(|buffer| buffers.get(buffer.index()).map(|data| &data.0[..]));
            let inverse_bind = reader
                .read_inverse_bind_matrices()
                .map(|iter| {
                    iter.map(|matrix| {
                        Mat4::from_column_slice(&[
                            matrix[0][0],
                            matrix[0][1],
                            matrix[0][2],
                            matrix[0][3],
                            matrix[1][0],
                            matrix[1][1],
                            matrix[1][2],
                            matrix[1][3],
                            matrix[2][0],
                            matrix[2][1],
                            matrix[2][2],
                            matrix[2][3],
                            matrix[3][0],
                            matrix[3][1],
                            matrix[3][2],
                            matrix[3][3],
                        ])
                    })
                    .collect()
                })
                .unwrap_or_default();
            GltfSkin {
                joints: skin.joints().map(|joint| joint.index()).collect(),
                inverse_bind,
            }
        })
        .collect();

    let animations = document
        .animations()
        .map(|animation| {
            let mut channels = Vec::new();
            let mut duration = 0.0f32;
            for channel in animation.channels() {
                let target_node = channel.target().node().index();
                let reader =
                    channel.reader(|buffer| buffers.get(buffer.index()).map(|data| &data.0[..]));
                let Some(input) = reader.read_inputs().map(|iter| iter.collect::<Vec<f32>>())
                else {
                    continue;
                };
                if let Some(&last) = input.last() {
                    duration = duration.max(last);
                }
                let keyframe_count = input.len();
                let Some(outputs) = reader.read_outputs() else {
                    continue;
                };
                let (property, output) = match outputs {
                    gltf::animation::util::ReadOutputs::Translations(iter) => (
                        AnimationProperty::Translation,
                        SamplerOutput::Vec3(
                            iter.map(|value| nalgebra_glm::vec3(value[0], value[1], value[2]))
                                .collect(),
                        ),
                    ),
                    gltf::animation::util::ReadOutputs::Scales(iter) => (
                        AnimationProperty::Scale,
                        SamplerOutput::Vec3(
                            iter.map(|value| nalgebra_glm::vec3(value[0], value[1], value[2]))
                                .collect(),
                        ),
                    ),
                    gltf::animation::util::ReadOutputs::Rotations(iter) => (
                        AnimationProperty::Rotation,
                        SamplerOutput::Quat(
                            iter.into_f32()
                                .map(|value| Quat::new(value[3], value[0], value[1], value[2]))
                                .collect(),
                        ),
                    ),
                    _ => continue,
                };
                let output = collapse_cubic_output(output, keyframe_count);
                let interpolation = match channel.sampler().interpolation() {
                    gltf::animation::Interpolation::Step => Interpolation::Step,
                    _ => Interpolation::Linear,
                };
                channels.push(AnimationChannel {
                    target_node,
                    property,
                    sampler: AnimationSampler {
                        input,
                        output,
                        interpolation,
                    },
                });
            }
            AnimationClip { duration, channels }
        })
        .collect();

    Some(GltfImport {
        primitives,
        skinned_primitives,
        base_color,
        normal,
        orm,
        emissive,
        base_factor: nalgebra_glm::vec3(base_factor[0], base_factor[1], base_factor[2]),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emissive_factor: nalgebra_glm::vec3(
            emissive_factor[0],
            emissive_factor[1],
            emissive_factor[2],
        ),
        nodes,
        skins,
        animations,
    })
}

fn collapse_cubic_output(output: SamplerOutput, keyframe_count: usize) -> SamplerOutput {
    match output {
        SamplerOutput::Vec3(values) if values.len() == keyframe_count * 3 => SamplerOutput::Vec3(
            (0..keyframe_count)
                .map(|index| values[index * 3 + 1])
                .collect(),
        ),
        SamplerOutput::Quat(values) if values.len() == keyframe_count * 3 => SamplerOutput::Quat(
            (0..keyframe_count)
                .map(|index| values[index * 3 + 1])
                .collect(),
        ),
        other => other,
    }
}

fn spawn_gltf(world: &mut World, scene: &GltfScene, placement: LocalTransform) {
    let mut root_mask = LOCAL_TRANSFORM | GLOBAL_TRANSFORM;
    if !scene.animations.is_empty() {
        root_mask |= ANIMATION_PLAYER;
    }
    let root = spawn(world, root_mask);
    if let Some(transform) = get_local_transform_mut(world, root) {
        *transform = placement;
    }
    mark_local_transform_dirty(world, root);
    let mut entities = Vec::with_capacity(scene.nodes.len());
    for node in &scene.nodes {
        let mut mask = LOCAL_TRANSFORM | GLOBAL_TRANSFORM | PARENT;
        if node.mesh.is_some() {
            mask |= MATERIAL | EMISSIVE;
            if node.skin.is_some() {
                mask |= SKINNED_MESH | SKIN;
            } else {
                mask |= RENDER_MESH;
            }
        }
        let entity = spawn(world, mask);
        if let Some(transform) = get_local_transform_mut(world, entity) {
            transform.translation = node.translation;
            transform.rotation = node.rotation;
            transform.scale = node.scale;
        }
        if let Some(mesh) = node.mesh {
            if node.skin.is_some() {
                if let Some(skinned_mesh) = get_skinned_mesh_mut(world, entity) {
                    skinned_mesh.0 = scene.skinned_mesh_base + mesh as u32;
                }
            } else if let Some(render_mesh) = get_render_mesh_mut(world, entity) {
                render_mesh.0 = scene.mesh_base + mesh as u32;
            }
            if let Some(material) = get_material_mut(world, entity) {
                *material = scene.material;
            }
            if let Some(emissive) = get_emissive_mut(world, entity) {
                emissive.0 = scene.emissive;
            }
        }
        entities.push(entity);
    }
    for (index, node) in scene.nodes.iter().enumerate() {
        let parent_entity = node.parent.map_or(root, |parent| entities[parent]);
        if let Some(parent) = component_mut(world, entities[index], PARENT, |table, row, tick| {
            table.parents[row].1 = tick;
            &mut table.parents[row].0
        }) {
            parent.0 = Some(parent_entity);
        }
        mark_local_transform_dirty(world, entities[index]);
    }
    for (index, node) in scene.nodes.iter().enumerate() {
        if let Some(skin_index) = node.skin
            && let Some(skin) = scene.skins.get(skin_index)
        {
            let joints: Vec<Entity> = skin
                .joints
                .iter()
                .map(|&joint_node| entities[joint_node])
                .collect();
            if let Some(skin_component) = get_skin_mut(world, entities[index]) {
                skin_component.joints = joints;
                skin_component.inverse_bind = skin.inverse_bind.clone();
            }
        }
    }
    if !scene.animations.is_empty() {
        let node_map: HashMap<usize, Entity> = entities
            .iter()
            .enumerate()
            .map(|(index, &entity)| (index, entity))
            .collect();
        if let Some(player) = get_animation_player_mut(world, root) {
            player.clips = scene.animations.clone();
            player.current_clip = Some(0);
            player.node_index_to_entity = node_map;
        }
    }
}

fn with_uv(vertices: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(vertices.len() / 9 * 11);
    for vertex in vertices.chunks_exact(9) {
        out.extend_from_slice(vertex);
        out.extend_from_slice(&[0.0, 0.0]);
    }
    out
}

#[cfg(target_arch = "wasm32")]
fn fetch_hdri(_url: &str) -> Option<(Vec<f32>, u32, u32)> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_hdri(url: &str) -> Option<(Vec<f32>, u32, u32)> {
    use std::io::Read;
    let cache = std::env::temp_dir().join("engine_codegolfed_hdri.hdr");
    let bytes = match std::fs::read(&cache) {
        Ok(bytes) => bytes,
        Err(_) => {
            let mut bytes = Vec::new();
            ureq::get(url)
                .call()
                .ok()?
                .into_reader()
                .read_to_end(&mut bytes)
                .ok()?;
            let _ = std::fs::write(&cache, &bytes);
            bytes
        }
    };
    let decoded = image::load_from_memory(&bytes).ok()?.to_rgba32f();
    let (width, height) = decoded.dimensions();
    Some((decoded.into_raw(), width, height))
}

fn equirect_texture(device: &wgpu::Device, queue: &wgpu::Queue, url: &str) -> wgpu::TextureView {
    let (data, width, height) =
        fetch_hdri(url).unwrap_or_else(|| (vec![0.4, 0.4, 0.45, 1.0], 1, 1));
    let half_data: Vec<u16> = data
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect();
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&half_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        extent,
    );
    texture.create_view(&Default::default())
}

fn frustum_planes(view_projection: &Mat4) -> [f32; 24] {
    let row = |index: usize| {
        [
            view_projection[(index, 0)],
            view_projection[(index, 1)],
            view_projection[(index, 2)],
            view_projection[(index, 3)],
        ]
    };
    let r0 = row(0);
    let r1 = row(1);
    let r3 = row(3);
    let sides = [
        [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]],
        [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]],
        [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]],
        [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]],
    ];
    let mut out = [0.0f32; 24];
    for (index, plane) in sides.iter().enumerate() {
        let length = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2])
            .sqrt()
            .max(1e-6);
        out[index * 4] = plane[0] / length;
        out[index * 4 + 1] = plane[1] / length;
        out[index * 4 + 2] = plane[2] / length;
        out[index * 4 + 3] = plane[3] / length;
    }
    out[19] = 1.0e9;
    out[23] = 1.0e9;
    out
}

fn mesh_gpu(device: &wgpu::Device, queue: &wgpu::Queue, vertices: &[f32]) -> MeshGpu {
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of_val(vertices) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(vertices));
    let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 176,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let visible_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut minimum = [f32::MAX; 3];
    let mut maximum = [f32::MIN; 3];
    for vertex in vertices.chunks_exact(11) {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(vertex[axis]);
            maximum[axis] = maximum[axis].max(vertex[axis]);
        }
    }
    let center = nalgebra_glm::vec3(
        (minimum[0] + maximum[0]) * 0.5,
        (minimum[1] + maximum[1]) * 0.5,
        (minimum[2] + maximum[2]) * 0.5,
    );
    let radius = nalgebra_glm::vec3(
        maximum[0] - center.x,
        maximum[1] - center.y,
        maximum[2] - center.z,
    )
    .norm();
    let bounds_buffer = uniform_buffer(device, 16);
    queue.write_buffer(
        &bounds_buffer,
        0,
        bytemuck::cast_slice(&[center.x, center.y, center.z, radius]),
    );
    let cull_buffer = uniform_buffer(device, 112);
    MeshGpu {
        vertex_buffer,
        vertex_count: vertices.len() as u32 / 11,
        instance_buffer,
        instance_capacity: 1,
        visible_indices_buffer,
        indirect_buffer,
        bounds_buffer,
        cull_buffer,
        objects_bind_group: None,
        shadow_bind_group: None,
        point_bind_group: None,
        cached_instances: Vec::new(),
    }
}

struct SkinnedGpu {
    rest_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    vertex_count: u32,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    visible_indices_buffer: wgpu::Buffer,
    objects_bind_group: Option<wgpu::BindGroup>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    point_bind_group: Option<wgpu::BindGroup>,
}

fn skinned_gpu(device: &wgpu::Device, queue: &wgpu::Queue, vertices: &[u32]) -> SkinnedGpu {
    let vertex_count = vertices.len() as u32 / 16;
    let rest_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of_val(vertices).max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&rest_buffer, 0, bytemuck::cast_slice(vertices));
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (vertex_count as u64 * 44).max(44),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_buffer = uniform_buffer(device, 16);
    let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 176,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let visible_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&visible_indices_buffer, 0, bytemuck::cast_slice(&[0u32]));
    SkinnedGpu {
        rest_buffer,
        output_buffer,
        params_buffer,
        vertex_count,
        instance_buffer,
        instance_capacity: 1,
        visible_indices_buffer,
        objects_bind_group: None,
        shadow_bind_group: None,
        point_bind_group: None,
    }
}

impl GeometryPass {
    fn render_shadow_atlas(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        bind_groups: &[wgpu::BindGroup],
        count: usize,
        counts: &[u32],
        skinned_draws: &[(usize, u32)],
    ) {
        for (slot, bind_group) in bind_groups.iter().take(count).enumerate() {
            let load = if slot == 0 {
                wgpu::LoadOp::Clear(0.0)
            } else {
                wgpu::LoadOp::Load
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            let slot_x = (slot % 2) as f32 * 1024.0;
            let slot_y = (slot / 2) as f32 * 1024.0;
            pass.set_viewport(slot_x, slot_y, 1024.0, 1024.0, 0.0, 1.0);
            pass.set_scissor_rect(slot_x as u32, slot_y as u32, 1024, 1024);
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            for (handle, mesh) in self.meshes.iter().enumerate() {
                if counts[handle] > 0
                    && let Some(group) = &mesh.shadow_bind_group
                {
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_bind_group(1, group, &[]);
                    pass.draw(0..mesh.vertex_count, 0..counts[handle]);
                }
            }
            for &(handle, vertex_count) in skinned_draws {
                let skinned = &self.skinned[handle];
                if let Some(group) = &skinned.shadow_bind_group {
                    pass.set_vertex_buffer(0, skinned.output_buffer.slice(..));
                    pass.set_bind_group(1, group, &[]);
                    pass.draw(0..vertex_count, 0..1);
                }
            }
        }
    }
}

impl PassNode for GeometryPass {
    fn color_writes(&self) -> Vec<&'static str> {
        vec!["color", "normals"]
    }

    fn depth_write(&self) -> Option<&'static str> {
        Some("depth")
    }

    fn execute(&mut self, context: &mut PassContext) {
        let view_projection = view_projection_matrix(context.world, context.aspect_ratio)
            .unwrap_or_else(Mat4::identity);
        let view = camera_view(context.world).unwrap_or_else(Mat4::identity);

        let projection =
            camera_projection(context.world, context.aspect_ratio).unwrap_or_else(Mat4::identity);
        let inverse_projection = nalgebra_glm::inverse(&projection);
        let (lights, spot_shadows) = gather_lights(context.world);
        let sun = nalgebra_glm::vec3(lights[4], lights[5], lights[6]);
        let camera_position = view
            .try_inverse()
            .map_or(Vec3::zeros(), |inverse| transform_translation(&inverse));
        let mut camera_data = [0.0f32; 52];
        camera_data[..16].copy_from_slice(view_projection.as_slice());
        camera_data[16..32].copy_from_slice(view.as_slice());
        camera_data[32..35].copy_from_slice(camera_position.as_slice());
        camera_data[36..52].copy_from_slice(inverse_projection.as_slice());
        context
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&camera_data));

        let (near, y_fov) = camera_near_fov(context.world);
        let splits = cascade_splits(near);
        let light_travel = -sun;
        let mut shadow_data = [0.0f32; 92];
        for cascade in 0..4 {
            let cascade_near = if cascade == 0 {
                near
            } else {
                splits[cascade - 1]
            };
            let corners = frustum_corners_world(
                &view,
                y_fov,
                context.aspect_ratio,
                cascade_near,
                splits[cascade],
            );
            let (cascade_vp, texel_world) = cascade_view_projection(&corners, light_travel, 1024.0);
            shadow_data[cascade * 16..cascade * 16 + 16].copy_from_slice(cascade_vp.as_slice());
            shadow_data[68 + cascade * 4] = (cascade % 2) as f32 * 0.5;
            shadow_data[68 + cascade * 4 + 1] = (cascade / 2) as f32 * 0.5;
            shadow_data[68 + cascade * 4 + 2] = texel_world;
            context.queue.write_buffer(
                &self.cascade_buffers[cascade],
                0,
                bytemuck::cast_slice(cascade_vp.as_slice()),
            );
        }
        shadow_data[64..68].copy_from_slice(&splits);
        shadow_data[84] = 0.5;
        shadow_data[88] = 1.0;
        shadow_data[89] = 2.0;
        shadow_data[90] = 0.0008;
        shadow_data[91] = 1.0;
        context
            .queue
            .write_buffer(&self.shadow_buffer, 0, bytemuck::cast_slice(&shadow_data));
        context
            .queue
            .write_buffer(&self.lights_buffer, 0, bytemuck::cast_slice(&lights));

        let mut spot_shadow_data = [0.0f32; 80];
        for (slot, spot_vp) in spot_shadows.iter().enumerate() {
            spot_shadow_data[slot * 16..slot * 16 + 16].copy_from_slice(spot_vp.as_slice());
            spot_shadow_data[64 + slot * 4] = (slot % 2) as f32 * 0.5;
            spot_shadow_data[64 + slot * 4 + 1] = (slot / 2) as f32 * 0.5;
            spot_shadow_data[64 + slot * 4 + 2] = 0.5;
            spot_shadow_data[64 + slot * 4 + 3] = 0.0008;
            context.queue.write_buffer(
                &self.spot_buffers[slot],
                0,
                bytemuck::cast_slice(spot_vp.as_slice()),
            );
        }
        context.queue.write_buffer(
            &self.spot_shadow_buffer,
            0,
            bytemuck::cast_slice(&spot_shadow_data),
        );

        let screen_width = context.size.0.max(1) as f32;
        let screen_height = context.size.1.max(1) as f32;
        let point_count = lights[12] as u32;
        let mut cluster_lights = [0.0f32; 64];
        for index in 0..point_count as usize {
            cluster_lights[index * 8..index * 8 + 4]
                .copy_from_slice(&lights[16 + index * 4..16 + index * 4 + 4]);
            cluster_lights[index * 8 + 4..index * 8 + 8]
                .copy_from_slice(&lights[80 + index * 4..80 + index * 4 + 4]);
        }
        context.queue.write_buffer(
            &self.cluster_lights_buffer,
            0,
            bytemuck::cast_slice(&cluster_lights),
        );
        let mut cluster_data = [0u32; 44];
        for (index, value) in inverse_projection.as_slice().iter().enumerate() {
            cluster_data[index] = value.to_bits();
        }
        for (index, value) in view.as_slice().iter().enumerate() {
            cluster_data[16 + index] = value.to_bits();
        }
        cluster_data[32] = screen_width.to_bits();
        cluster_data[33] = screen_height.to_bits();
        cluster_data[34] = 0.1f32.to_bits();
        cluster_data[35] = 100.0f32.to_bits();
        cluster_data[36] = CLUSTER_X;
        cluster_data[37] = CLUSTER_Y;
        cluster_data[38] = CLUSTER_Z;
        cluster_data[39] = MAX_LIGHTS_PER_CLUSTER;
        cluster_data[40] = (screen_width / CLUSTER_X as f32).to_bits();
        cluster_data[41] = (screen_height / CLUSTER_Y as f32).to_bits();
        cluster_data[42] = point_count;
        context.queue.write_buffer(
            &self.cluster_uniform_buffer,
            0,
            bytemuck::cast_slice(&cluster_data),
        );
        {
            let mut compute = context
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            compute.set_pipeline(&self.cluster_bounds_pipeline);
            compute.set_bind_group(0, &self.cluster_bounds_bind_group, &[]);
            compute.dispatch_workgroups(4, 3, 6);
        }
        {
            let mut compute = context
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            compute.set_pipeline(&self.cluster_assign_pipeline);
            compute.set_bind_group(0, &self.cluster_assign_bind_group, &[]);
            compute.dispatch_workgroups(4, 3, 6);
        }

        let frustum = frustum_planes(
            &view_projection_matrix(context.world, context.aspect_ratio)
                .unwrap_or_else(Mat4::identity),
        );
        let mut counts = Vec::with_capacity(self.meshes.len());
        for (handle, mesh) in self.meshes.iter_mut().enumerate() {
            let instances = mesh_instances(context.world, handle as u32);
            let count = instances.len() as u32;
            let mut data: Vec<f32> = Vec::with_capacity(instances.len() * 44);
            for (model, emissive, material) in &instances {
                data.extend_from_slice(model.as_slice());
                data.extend_from_slice(&[0.0; 12]);
                data.extend_from_slice(&[emissive.x, emissive.y, emissive.z, material.roughness]);
                data.extend_from_slice(&[
                    material.albedo.x,
                    material.albedo.y,
                    material.albedo.z,
                    material.metallic,
                ]);
                data.push(material.base_layer as f32);
                data.push(material.normal_layer as f32);
                data.push(material.orm_layer as f32);
                data.push(material.emissive_layer as f32);
                data.extend_from_slice(&[f32::from_bits(1), 0.0, 0.0, 0.0]);
            }
            upload_instances_diff(
                context.device,
                context.queue,
                &mut mesh.instance_buffer,
                &mut mesh.instance_capacity,
                &mut mesh.cached_instances,
                &data,
                count,
            );
            counts.push(count);
            if count == 0 {
                continue;
            }
            let needed = mesh.instance_capacity as u64 * 4;
            if mesh.visible_indices_buffer.size() < needed {
                mesh.visible_indices_buffer =
                    context.device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: needed,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
            }
            context.queue.write_buffer(
                &mesh.indirect_buffer,
                0,
                bytemuck::cast_slice(&[mesh.vertex_count, 0u32, 0, 0]),
            );
            let mut cull_data = [0u32; 28];
            for (index, value) in frustum.iter().enumerate() {
                cull_data[index] = value.to_bits();
            }
            cull_data[24] = count;
            context
                .queue
                .write_buffer(&mesh.cull_buffer, 0, bytemuck::cast_slice(&cull_data));
            let cull_bind = bind_group(
                context.device,
                &self.cull_pipeline.get_bind_group_layout(0),
                vec![
                    (0, mesh.instance_buffer.as_entire_binding()),
                    (1, mesh.visible_indices_buffer.as_entire_binding()),
                    (2, mesh.indirect_buffer.as_entire_binding()),
                    (3, mesh.bounds_buffer.as_entire_binding()),
                    (4, mesh.cull_buffer.as_entire_binding()),
                ],
            );
            mesh.objects_bind_group = Some(bind_group(
                context.device,
                &self.pipeline.get_bind_group_layout(1),
                vec![
                    (0, mesh.instance_buffer.as_entire_binding()),
                    (1, mesh.visible_indices_buffer.as_entire_binding()),
                ],
            ));
            mesh.shadow_bind_group = Some(bind_group(
                context.device,
                &self.shadow_pipeline.get_bind_group_layout(1),
                vec![(0, mesh.instance_buffer.as_entire_binding())],
            ));
            mesh.point_bind_group = Some(bind_group(
                context.device,
                &self.point_pipeline.get_bind_group_layout(1),
                vec![(0, mesh.instance_buffer.as_entire_binding())],
            ));
            let mut cull = context
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            cull.set_pipeline(&self.cull_pipeline);
            cull.set_bind_group(0, &cull_bind, &[]);
            cull.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }

        let skinned = skinned_instances(context.world);
        let mut joint_data: Vec<f32> = Vec::new();
        let mut skinned_draws: Vec<(usize, u32)> = Vec::new();
        for (handle, matrices, emissive, material) in &skinned {
            let handle = *handle as usize;
            if handle >= self.skinned.len() {
                continue;
            }
            let joint_offset = (joint_data.len() / 16) as u32;
            for matrix in matrices {
                joint_data.extend_from_slice(matrix.as_slice());
            }
            let vertex_count = self.skinned[handle].vertex_count;
            context.queue.write_buffer(
                &self.skinned[handle].params_buffer,
                0,
                bytemuck::cast_slice(&[vertex_count, joint_offset, 0, 0]),
            );
            let mut instance: Vec<f32> = Vec::with_capacity(44);
            instance.extend_from_slice(Mat4::identity().as_slice());
            instance
                .extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
            instance.extend_from_slice(&[emissive.x, emissive.y, emissive.z, material.roughness]);
            instance.extend_from_slice(&[
                material.albedo.x,
                material.albedo.y,
                material.albedo.z,
                material.metallic,
            ]);
            instance.push(material.base_layer as f32);
            instance.push(material.normal_layer as f32);
            instance.push(material.orm_layer as f32);
            instance.push(material.emissive_layer as f32);
            instance.extend_from_slice(&[f32::from_bits(1), 0.0, 0.0, 0.0]);
            let target = &mut self.skinned[handle];
            upload_instances(
                context.device,
                context.queue,
                &mut target.instance_buffer,
                &mut target.instance_capacity,
                &instance,
                1,
            );
            target.objects_bind_group = Some(bind_group(
                context.device,
                &self.pipeline.get_bind_group_layout(1),
                vec![
                    (0, target.instance_buffer.as_entire_binding()),
                    (1, target.visible_indices_buffer.as_entire_binding()),
                ],
            ));
            target.shadow_bind_group = Some(bind_group(
                context.device,
                &self.shadow_pipeline.get_bind_group_layout(1),
                vec![(0, target.instance_buffer.as_entire_binding())],
            ));
            target.point_bind_group = Some(bind_group(
                context.device,
                &self.point_pipeline.get_bind_group_layout(1),
                vec![(0, target.instance_buffer.as_entire_binding())],
            ));
            skinned_draws.push((handle, vertex_count));
        }
        if !joint_data.is_empty() {
            let needed = joint_data.len() as u32;
            if needed > self.joint_capacity {
                self.joint_capacity = needed.next_power_of_two();
                self.joint_buffer = storage_buffer(context.device, self.joint_capacity as u64 * 4);
            }
            context
                .queue
                .write_buffer(&self.joint_buffer, 0, bytemuck::cast_slice(&joint_data));
            for &(handle, vertex_count) in &skinned_draws {
                let skinned = &self.skinned[handle];
                let bind = bind_group(
                    context.device,
                    &self.skin_pipeline.get_bind_group_layout(0),
                    vec![
                        (0, skinned.rest_buffer.as_entire_binding()),
                        (1, self.joint_buffer.as_entire_binding()),
                        (2, skinned.output_buffer.as_entire_binding()),
                        (3, skinned.params_buffer.as_entire_binding()),
                    ],
                );
                let mut compute =
                    context
                        .encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                compute.set_pipeline(&self.skin_pipeline);
                compute.set_bind_group(0, &bind, &[]);
                compute.dispatch_workgroups(vertex_count.div_ceil(64), 1, 1);
            }
        }

        self.render_shadow_atlas(
            context.encoder,
            &self.shadow_view,
            &self.cascade_bind_groups,
            4,
            &counts,
            &skinned_draws,
        );
        self.render_shadow_atlas(
            context.encoder,
            &self.spot_atlas_view,
            &self.spot_bind_groups,
            spot_shadows.len(),
            &counts,
            &skinned_draws,
        );

        let point_count = lights[12] as usize;
        let y_flip = Mat4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let mut shadow_slot = 0usize;
        for index in 0..point_count {
            if shadow_slot >= 4 || lights[80 + index * 4 + 3] >= -1.5 {
                continue;
            }
            let point_position = nalgebra_glm::vec3(
                lights[16 + index * 4],
                lights[16 + index * 4 + 1],
                lights[16 + index * 4 + 2],
            );
            let point_range = lights[16 + index * 4 + 3].max(0.5);
            let projection =
                y_flip * reverse_z_perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.05, point_range);
            for (face, (direction, up)) in CUBE_FACES.iter().enumerate() {
                let target = point_position + direction;
                let view_projection =
                    projection * nalgebra_glm::look_at(&point_position, &target, up);
                let mut face_data = [0.0f32; 20];
                face_data[..16].copy_from_slice(view_projection.as_slice());
                face_data[16..19].copy_from_slice(point_position.as_slice());
                face_data[19] = point_range;
                context.queue.write_buffer(
                    &self.point_face_buffers[shadow_slot * 6 + face],
                    0,
                    bytemuck::cast_slice(&face_data),
                );
            }
            for face in 0..6 {
                let layer = shadow_slot * 6 + face;
                let mut point_pass =
                    context
                        .encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &self.point_face_views[layer],
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &self.point_depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(0.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            ..Default::default()
                        });
                point_pass.set_pipeline(&self.point_pipeline);
                point_pass.set_bind_group(0, &self.point_face_bind_groups[layer], &[]);
                for (handle, mesh) in self.meshes.iter().enumerate() {
                    if counts[handle] > 0
                        && let Some(group) = &mesh.point_bind_group
                    {
                        point_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        point_pass.set_bind_group(1, group, &[]);
                        point_pass.draw(0..mesh.vertex_count, 0..counts[handle]);
                    }
                }
                for &(skinned_handle, vertex_count) in &skinned_draws {
                    let skinned = &self.skinned[skinned_handle];
                    if let Some(group) = &skinned.point_bind_group {
                        point_pass.set_vertex_buffer(0, skinned.output_buffer.slice(..));
                        point_pass.set_bind_group(1, group, &[]);
                        point_pass.draw(0..vertex_count, 0..1);
                    }
                }
            }
            shadow_slot += 1;
        }

        let (color_view, color_load, color_store) = color_attachment(context, "color");
        let (normal_view, normal_load, normal_store) = color_attachment(context, "normals");
        let (depth_view, depth_load, depth_store) = depth_attachment(context, "depth");
        let mut pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: color_load,
                            store: color_store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: normal_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: normal_load,
                            store: normal_store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: depth_load,
                        store: depth_store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        pass.set_pipeline(&self.sky_pipeline);
        pass.set_bind_group(0, &self.sky_bind_group, &[]);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        for (handle, mesh) in self.meshes.iter().enumerate() {
            if counts[handle] > 0
                && let Some(group) = &mesh.objects_bind_group
            {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_bind_group(1, group, &[]);
                pass.draw_indirect(&mesh.indirect_buffer, 0);
            }
        }
        for &(handle, vertex_count) in &skinned_draws {
            let skinned = &self.skinned[handle];
            if let Some(group) = &skinned.objects_bind_group {
                pass.set_vertex_buffer(0, skinned.output_buffer.slice(..));
                pass.set_bind_group(1, group, &[]);
                pass.draw(0..vertex_count, 0..1);
            }
        }
    }
}

enum Binding {
    Read(&'static str),
    Sampler,
    Uniform,
    Storage,
}

struct FullscreenPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    output: &'static str,
    bindings: Vec<Binding>,
    sampler: Option<wgpu::Sampler>,
    uniform: Option<wgpu::Buffer>,
    storage: Option<wgpu::Buffer>,
    update: Option<fn(&PassContext, &wgpu::Buffer)>,
}

fn ssao_update(context: &PassContext, buffer: &wgpu::Buffer) {
    let projection =
        camera_projection(context.world, context.aspect_ratio).unwrap_or_else(Mat4::identity);
    let inverse_projection = projection.try_inverse().unwrap_or_else(Mat4::identity);
    let mut data = [0.0f32; 20];
    data[..16].copy_from_slice(inverse_projection.as_slice());
    data[16..].copy_from_slice(&[24.0, 0.025, 2.5, 0.0]);
    context
        .queue
        .write_buffer(buffer, 0, bytemuck::cast_slice(&data));
}

fn composite_update(context: &PassContext, buffer: &wgpu::Buffer) {
    let bloom: f32 = if context.world.resources.bloom_enabled {
        0.8
    } else {
        0.0
    };
    context
        .queue
        .write_buffer(buffer, 0, bytemuck::cast_slice(&[bloom, 0.0, 0.0, 0.0]));
}

fn fxaa_update(context: &PassContext, buffer: &wgpu::Buffer) {
    let (width, height) = context.size;
    context.queue.write_buffer(
        buffer,
        0,
        bytemuck::cast_slice(&[
            1.0 / width.max(1) as f32,
            1.0 / height.max(1) as f32,
            1.0,
            0.75,
        ]),
    );
}

fn ssr_update(context: &PassContext, buffer: &wgpu::Buffer) {
    let projection =
        camera_projection(context.world, context.aspect_ratio).unwrap_or_else(Mat4::identity);
    let inverse = projection.try_inverse().unwrap_or_else(Mat4::identity);
    let (width, height) = context.size;
    let mut data = [0u32; 40];
    for (index, value) in projection.as_slice().iter().enumerate() {
        data[index] = value.to_bits();
    }
    for (index, value) in inverse.as_slice().iter().enumerate() {
        data[16 + index] = value.to_bits();
    }
    data[32] = (width as f32).to_bits();
    data[33] = (height as f32).to_bits();
    data[34] = 64;
    data[35] = 30.0f32.to_bits();
    data[36] = 0.7f32.to_bits();
    data[37] = 1.0f32.to_bits();
    data[38] = 0.7f32.to_bits();
    let enabled = if context.world.resources.ssr_enabled {
        1.0f32
    } else {
        0.0f32
    };
    data[39] = enabled.to_bits();
    context
        .queue
        .write_buffer(buffer, 0, bytemuck::cast_slice(&data));
}

fn ssr_blur_update(context: &PassContext, buffer: &wgpu::Buffer) {
    let (width, height) = context.size;
    context.queue.write_buffer(
        buffer,
        0,
        bytemuck::cast_slice(&[width as f32, height as f32, 0.001, 8.0]),
    );
}

impl PassNode for FullscreenPass {
    fn reads(&self) -> Vec<&'static str> {
        self.bindings
            .iter()
            .filter_map(|binding| match binding {
                Binding::Read(name) => Some(*name),
                _ => None,
            })
            .collect()
    }

    fn color_writes(&self) -> Vec<&'static str> {
        vec![self.output]
    }

    fn execute(&mut self, context: &mut PassContext) {
        if let (Some(update), Some(uniform)) = (self.update, &self.uniform) {
            update(context, uniform);
        }
        let entries: Vec<(u32, wgpu::BindingResource)> = self
            .bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| {
                let resource = match binding {
                    Binding::Read(name) => {
                        wgpu::BindingResource::TextureView(read_view(context, name))
                    }
                    Binding::Sampler => {
                        wgpu::BindingResource::Sampler(self.sampler.as_ref().unwrap())
                    }
                    Binding::Uniform => self.uniform.as_ref().unwrap().as_entire_binding(),
                    Binding::Storage => self.storage.as_ref().unwrap().as_entire_binding(),
                };
                (index as u32, resource)
            })
            .collect();
        let bind_group = bind_group(context.device, &self.bind_group_layout, entries);
        fullscreen_pass(context, &self.pipeline, &bind_group, self.output);
    }
}

struct AutoExposurePass {
    pipeline: wgpu::ComputePipeline,
    exposure_buffer: wgpu::Buffer,
}

impl PassNode for AutoExposurePass {
    fn reads(&self) -> Vec<&'static str> {
        vec!["scene"]
    }

    fn color_writes(&self) -> Vec<&'static str> {
        vec!["scene"]
    }

    fn execute(&mut self, context: &mut PassContext) {
        let delta_time = context.world.resources.delta_time.min(0.1);
        context.queue.write_buffer(
            &self.exposure_buffer,
            8,
            bytemuck::cast_slice(&[1.1f32, delta_time]),
        );
        let bind = bind_group(
            context.device,
            &self.pipeline.get_bind_group_layout(0),
            vec![
                (
                    0,
                    wgpu::BindingResource::TextureView(read_view(context, "scene")),
                ),
                (1, self.exposure_buffer.as_entire_binding()),
            ],
        );
        let mut compute = context
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        compute.set_pipeline(&self.pipeline);
        compute.set_bind_group(0, &bind, &[]);
        compute.dispatch_workgroups(1, 1, 1);
    }
}

#[cfg(target_arch = "wasm32")]
fn get_canvas() -> wgpu::web_sys::HtmlCanvasElement {
    wgpu::web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into()
        .unwrap()
}

struct Graphics {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    graph: RenderGraph,
    size: (u32, u32),
    gltf: Option<GltfScene>,
    gltf_skinned: Option<GltfScene>,
}

pub struct App {
    state: Box<dyn State>,
    world: World,
    graphics: Option<Graphics>,
    #[cfg(target_arch = "wasm32")]
    pending: Option<futures::channel::oneshot::Receiver<Graphics>>,
}

impl Default for App {
    fn default() -> Self {
        App::new(Box::new(Demo))
    }
}

impl App {
    pub fn new(state: Box<dyn State>) -> Self {
        Self {
            state,
            world: World::default(),
            graphics: None,
            #[cfg(target_arch = "wasm32")]
            pending: None,
        }
    }
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    fragment: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(format!("{FULLSCREEN_VERTEX}{fragment}").into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn linear_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

fn uniform_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn storage_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn compute_pipeline(device: &wgpu::Device, source: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn render_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    buffers: &[wgpu::VertexBufferLayout],
    targets: &[Option<wgpu::ColorTargetState>],
    depth_write: bool,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers,
        },
        fragment: (!targets.is_empty()).then(|| wgpu::FragmentState {
            module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets,
        }),
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(depth_compare),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

const TEXTURE_SIZE: usize = 512;
const TEXTURE_LAYERS: usize = 6;
const TEXTURE_CELL: usize = TEXTURE_SIZE / 8;

fn albedo_pixel(layer: usize, x: usize, y: usize) -> [u8; 4] {
    let checker = ((x / TEXTURE_CELL) + (y / TEXTURE_CELL)).is_multiple_of(2);
    let (r, g, b) = match layer {
        1 if checker => (230, 120, 40),
        1 => (60, 30, 15),
        2 if checker => (185, 185, 195),
        2 => (120, 120, 130),
        3 if checker => (40, 120, 200),
        3 => (15, 30, 60),
        _ => (255, 255, 255),
    };
    [r, g, b, 255]
}

fn normal_pixel(layer: usize, x: usize, y: usize) -> [u8; 4] {
    let strength = match layer {
        1 => 0.7,
        2 => 0.5,
        3 => 0.9,
        _ => 0.0,
    };
    let cell_u = ((x % TEXTURE_CELL) as f32 / TEXTURE_CELL as f32 * 2.0 - 1.0) * strength;
    let cell_v = ((y % TEXTURE_CELL) as f32 / TEXTURE_CELL as f32 * 2.0 - 1.0) * strength;
    let length = (cell_u * cell_u + cell_v * cell_v + 1.0).sqrt();
    let encode = |value: f32| ((value / length * 0.5 + 0.5) * 255.0) as u8;
    [encode(cell_u), encode(cell_v), encode(1.0), 255]
}

fn orm_pixel(layer: usize, x: usize, y: usize) -> [u8; 4] {
    let checker = ((x / TEXTURE_CELL) + (y / TEXTURE_CELL)).is_multiple_of(2);
    let roughness = match layer {
        1 if checker => 255,
        1 => 70,
        _ => 255,
    };
    [255, roughness, 255, 255]
}

fn emissive_pixel(_layer: usize, _x: usize, _y: usize) -> [u8; 4] {
    [255, 255, 255, 255]
}

fn texture_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    fill: impl Fn(usize, usize, usize) -> [u8; 4],
) -> wgpu::TextureView {
    let size = TEXTURE_SIZE;
    let layers = TEXTURE_LAYERS;
    let mut pixels = vec![0u8; size * size * layers * 4];
    for layer in 0..layers {
        for y in 0..size {
            for x in 0..size {
                let index = (layer * size * size + y * size + x) * 4;
                pixels[index..index + 4].copy_from_slice(&fill(layer, x, y));
            }
        }
    }
    let extent = wgpu::Extent3d {
        width: size as u32,
        height: size as u32,
        depth_or_array_layers: layers as u32,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size as u32 * 4),
            rows_per_image: Some(size as u32),
        },
        extent,
    );
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn shadow_atlas_view(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 2048,
                height: 2048,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn ibl_storage_texture(device: &wgpu::Device, size: u32, layers: u32, mips: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn filter_params(
    output_size: u32,
    roughness: f32,
    samples: u32,
    distribution: u32,
    mips: u32,
) -> [u32; 8] {
    [
        0,
        output_size,
        roughness.to_bits(),
        samples,
        2048,
        2048,
        distribution,
        mips,
    ]
}

fn array_mip_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_mip_level: mip,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

fn cube_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

fn dispatch_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    groups: (u32, u32, u32),
) {
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
    queue.submit([encoder.finish()]);
}

fn generate_ibl(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    equirect: &wgpu::TextureView,
) -> (
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::Sampler,
) {
    let env_mips = 12u32;
    let env = ibl_storage_texture(device, 2048, 6, env_mips);
    let irradiance = ibl_storage_texture(device, 64, 6, 1);
    let prefiltered = ibl_storage_texture(device, 512, 6, 5);
    let brdf = ibl_storage_texture(device, 256, 1, 1);

    let equirect_pipeline = compute_pipeline(device, EQUIRECT_TO_CUBE_SHADER);
    let mipgen_pipeline = compute_pipeline(device, CUBEMAP_MIPGEN_SHADER);
    let filter_pipeline = compute_pipeline(device, FILTER_ENVMAP_SHADER);
    let brdf_pipeline = compute_pipeline(device, &format!("{IBL_COMMON}{BRDF_LUT_SHADER}"));

    let equirect_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let env_storage = array_mip_view(&env, 0);
    let equirect_bind = bind_group(
        device,
        &equirect_pipeline.get_bind_group_layout(0),
        vec![
            (0, wgpu::BindingResource::TextureView(equirect)),
            (1, wgpu::BindingResource::Sampler(&equirect_sampler)),
            (2, wgpu::BindingResource::TextureView(&env_storage)),
        ],
    );
    dispatch_compute(
        device,
        queue,
        &equirect_pipeline,
        &equirect_bind,
        (128, 128, 6),
    );

    for mip in 1..env_mips {
        let size = 2048u32 >> mip;
        let source = array_mip_view(&env, mip - 1);
        let destination = array_mip_view(&env, mip);
        let params = uniform_buffer(device, 16);
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&[size, 0u32, 0, 0]));
        let bind = bind_group(
            device,
            &mipgen_pipeline.get_bind_group_layout(0),
            vec![
                (0, wgpu::BindingResource::TextureView(&source)),
                (1, wgpu::BindingResource::Sampler(&sampler)),
                (2, wgpu::BindingResource::TextureView(&destination)),
                (3, params.as_entire_binding()),
            ],
        );
        dispatch_compute(
            device,
            queue,
            &mipgen_pipeline,
            &bind,
            (size.div_ceil(16), size.div_ceil(16), 6),
        );
    }

    let env_cube = cube_view(&env);

    let irradiance_storage = array_mip_view(&irradiance, 0);
    let irradiance_params = uniform_buffer(device, 32);
    queue.write_buffer(
        &irradiance_params,
        0,
        bytemuck::cast_slice(&filter_params(64, 1.0, 1024, 0, env_mips)),
    );
    let irradiance_bind = bind_group(
        device,
        &filter_pipeline.get_bind_group_layout(0),
        vec![
            (0, wgpu::BindingResource::TextureView(&env_cube)),
            (1, wgpu::BindingResource::Sampler(&sampler)),
            (2, wgpu::BindingResource::TextureView(&irradiance_storage)),
            (3, irradiance_params.as_entire_binding()),
        ],
    );
    dispatch_compute(device, queue, &filter_pipeline, &irradiance_bind, (4, 4, 6));

    for mip in 0..5u32 {
        let size = 512u32 >> mip;
        let storage = array_mip_view(&prefiltered, mip);
        let params = uniform_buffer(device, 32);
        queue.write_buffer(
            &params,
            0,
            bytemuck::cast_slice(&filter_params(size, mip as f32 / 4.0, 512, 1, env_mips)),
        );
        let bind = bind_group(
            device,
            &filter_pipeline.get_bind_group_layout(0),
            vec![
                (0, wgpu::BindingResource::TextureView(&env_cube)),
                (1, wgpu::BindingResource::Sampler(&sampler)),
                (2, wgpu::BindingResource::TextureView(&storage)),
                (3, params.as_entire_binding()),
            ],
        );
        dispatch_compute(
            device,
            queue,
            &filter_pipeline,
            &bind,
            (size.div_ceil(16), size.div_ceil(16), 6),
        );
    }

    let brdf_storage = brdf.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    });
    let brdf_bind = bind_group(
        device,
        &brdf_pipeline.get_bind_group_layout(0),
        vec![(0, wgpu::BindingResource::TextureView(&brdf_storage))],
    );
    dispatch_compute(device, queue, &brdf_pipeline, &brdf_bind, (32, 32, 1));

    let brdf_view = brdf.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    });
    (
        env_cube,
        cube_view(&irradiance),
        cube_view(&prefiltered),
        brdf_view,
        sampler,
    )
}

fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    entries: Vec<(u32, wgpu::BindingResource)>,
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = entries
        .into_iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

fn axis_buffer(device: &wgpu::Device, queue: &wgpu::Queue, axis: [f32; 4]) -> wgpu::Buffer {
    let buffer = uniform_buffer(device, 16);
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&axis));
    buffer
}

async fn init_graphics(window: Arc<Window>, width: u32, height: u32) -> Graphics {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            ..Default::default()
        })
        .await
        .unwrap();
    let surface_config = surface.get_default_config(&adapter, width, height).unwrap();
    surface.configure(&device, &surface_config);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let vertex_attrs =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 7 => Float32x3, 13 => Float32x2];
    let mesh_color_buffers = [wgpu::VertexBufferLayout {
        array_stride: 44,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attrs,
    }];
    let pipeline = render_pipeline(
        &device,
        &shader,
        &mesh_color_buffers,
        &[Some(SCENE_FORMAT.into()), Some(SCENE_FORMAT.into())],
        true,
        wgpu::CompareFunction::Greater,
    );

    let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
    });
    let shadow_pipeline = render_pipeline(
        &device,
        &shadow_shader,
        &mesh_color_buffers,
        &[],
        true,
        wgpu::CompareFunction::GreaterEqual,
    );
    let shadow_view = shadow_atlas_view(&device);
    let spot_atlas_view = shadow_atlas_view(&device);
    let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bright_pipeline = fullscreen_pipeline(&device, BRIGHT_SHADER, SCENE_FORMAT);
    let bright_bind_group_layout = bright_pipeline.get_bind_group_layout(0);

    let blur_horizontal_pipeline = fullscreen_pipeline(&device, BLUR_SHADER, SCENE_FORMAT);
    let blur_horizontal_bind_group_layout = blur_horizontal_pipeline.get_bind_group_layout(0);
    let blur_vertical_pipeline = fullscreen_pipeline(&device, BLUR_SHADER, SCENE_FORMAT);
    let blur_vertical_bind_group_layout = blur_vertical_pipeline.get_bind_group_layout(0);

    let composite_pipeline = fullscreen_pipeline(&device, COMPOSITE_SHADER, SCENE_FORMAT);
    let composite_bind_group_layout = composite_pipeline.get_bind_group_layout(0);
    let composite_params_buffer = uniform_buffer(&device, 16);
    let fxaa_pipeline = fullscreen_pipeline(&device, FXAA_SHADER, surface_config.format);
    let fxaa_bind_group_layout = fxaa_pipeline.get_bind_group_layout(0);
    let fxaa_params_buffer = uniform_buffer(&device, 16);
    let auto_exposure_pipeline = compute_pipeline(&device, AUTO_EXPOSURE_SHADER);
    let exposure_buffer = storage_buffer(&device, 32);
    let ssr_pipeline = fullscreen_pipeline(&device, SSR_SHADER, SCENE_FORMAT);
    let ssr_bind_group_layout = ssr_pipeline.get_bind_group_layout(0);
    let ssr_params_buffer = uniform_buffer(&device, 160);
    let ssr_blur_pipeline = fullscreen_pipeline(&device, SSR_BLUR_SHADER, SCENE_FORMAT);
    let ssr_blur_bind_group_layout = ssr_blur_pipeline.get_bind_group_layout(0);
    let ssr_blur_params_buffer = uniform_buffer(&device, 16);

    let helmet = fetch_gltf(
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb",
    );
    let character = fetch_gltf(
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb",
    );
    let mut meshes = vec![
        mesh_gpu(&device, &queue, &with_uv(&CUBE_VERTICES)),
        mesh_gpu(&device, &queue, &sphere_mesh(16, 32)),
    ];
    if let Some(ref import) = helmet {
        for primitive in &import.primitives {
            meshes.push(mesh_gpu(&device, &queue, primitive));
        }
    }
    let mut skinned = Vec::new();
    if let Some(ref import) = character {
        for primitive in &import.skinned_primitives {
            skinned.push(skinned_gpu(&device, &queue, primitive));
        }
    }

    let camera_buffer = uniform_buffer(&device, 208);
    let lights_buffer = uniform_buffer(&device, 576);
    let shadow_buffer = uniform_buffer(&device, 368);
    let spot_shadow_buffer = uniform_buffer(&device, 320);
    let cluster_uniform_buffer = uniform_buffer(&device, 176);
    let cluster_bounds_buffer = storage_buffer(&device, CLUSTER_TOTAL as u64 * 32);
    let light_grid_buffer = storage_buffer(&device, CLUSTER_TOTAL as u64 * 8);
    let light_indices_buffer = storage_buffer(
        &device,
        CLUSTER_TOTAL as u64 * MAX_LIGHTS_PER_CLUSTER as u64 * 4,
    );
    let cluster_lights_buffer = storage_buffer(&device, 8 * 32);
    let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let helmet_layer = |source: &Option<Vec<u8>>, x: usize, y: usize, default: [u8; 4]| {
        source.as_ref().map_or(default, |pixels| {
            let index = (y * TEXTURE_SIZE + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        })
    };
    let albedo_array_view = texture_array(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        |layer, x, y| match (&helmet, &character, layer) {
            (Some(mesh), _, 4) => helmet_layer(&mesh.base_color, x, y, [255, 255, 255, 255]),
            (_, Some(mesh), 5) => helmet_layer(&mesh.base_color, x, y, [255, 255, 255, 255]),
            _ => albedo_pixel(layer, x, y),
        },
    );
    let normal_array_view = texture_array(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8Unorm,
        |layer, x, y| match (&helmet, &character, layer) {
            (Some(mesh), _, 4) => helmet_layer(&mesh.normal, x, y, [128, 128, 255, 255]),
            (_, Some(mesh), 5) => helmet_layer(&mesh.normal, x, y, [128, 128, 255, 255]),
            _ => normal_pixel(layer, x, y),
        },
    );
    let orm_array_view = texture_array(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8Unorm,
        |layer, x, y| match (&helmet, &character, layer) {
            (Some(mesh), _, 4) => helmet_layer(&mesh.orm, x, y, [255, 255, 255, 255]),
            (_, Some(mesh), 5) => helmet_layer(&mesh.orm, x, y, [255, 255, 255, 255]),
            _ => orm_pixel(layer, x, y),
        },
    );
    let emissive_array_view = texture_array(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        |layer, x, y| match (&helmet, &character, layer) {
            (Some(mesh), _, 4) => helmet_layer(&mesh.emissive, x, y, [0, 0, 0, 255]),
            (_, Some(mesh), 5) => helmet_layer(&mesh.emissive, x, y, [0, 0, 0, 255]),
            _ => emissive_pixel(layer, x, y),
        },
    );
    let gltf_scene = helmet.map(|import| GltfScene {
        nodes: import.nodes,
        material: Material {
            albedo: import.base_factor,
            metallic: import.metallic,
            roughness: import.roughness,
            base_layer: 4,
            normal_layer: 4,
            orm_layer: 4,
            emissive_layer: 4,
        },
        emissive: import.emissive_factor,
        mesh_base: 2,
        skinned_mesh_base: 0,
        skins: import.skins,
        animations: import.animations,
    });
    let character_scene = character.map(|import| GltfScene {
        nodes: import.nodes,
        material: Material {
            albedo: import.base_factor,
            metallic: import.metallic,
            roughness: import.roughness,
            base_layer: 5,
            normal_layer: 5,
            orm_layer: 5,
            emissive_layer: 5,
        },
        emissive: import.emissive_factor,
        mesh_base: 2,
        skinned_mesh_base: 0,
        skins: import.skins,
        animations: import.animations,
    });
    let equirect_view = equirect_texture(
        &device,
        &queue,
        "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/symmetrical_garden_02_2k.hdr",
    );
    let (env_cube_view, irradiance_view, prefiltered_view, brdf_view, ibl_sampler) =
        generate_ibl(&device, &queue, &equirect_view);
    let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(POINT_SHADOW_SHADER.into()),
    });
    let point_pipeline = render_pipeline(
        &device,
        &point_shader,
        &mesh_color_buffers,
        &[Some(wgpu::TextureFormat::R16Float.into())],
        true,
        wgpu::CompareFunction::GreaterEqual,
    );
    let point_cube_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 24,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let point_cube_view = point_cube_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::CubeArray),
        ..Default::default()
    });
    let point_face_views: [wgpu::TextureView; 24] = std::array::from_fn(|layer| {
        point_cube_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer as u32,
            array_layer_count: Some(1),
            ..Default::default()
        })
    });
    let point_depth_view = device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let point_face_buffers: [wgpu::Buffer; 24] =
        std::array::from_fn(|_| uniform_buffer(&device, 80));
    let point_face_bind_groups: [wgpu::BindGroup; 24] = std::array::from_fn(|layer| {
        bind_group(
            &device,
            &point_pipeline.get_bind_group_layout(0),
            vec![(0, point_face_buffers[layer].as_entire_binding())],
        )
    });
    let geometry_bind_group = bind_group(
        &device,
        &pipeline.get_bind_group_layout(0),
        vec![
            (0, camera_buffer.as_entire_binding()),
            (1, lights_buffer.as_entire_binding()),
            (2, wgpu::BindingResource::TextureView(&shadow_view)),
            (3, wgpu::BindingResource::Sampler(&shadow_sampler)),
            (4, shadow_buffer.as_entire_binding()),
            (5, wgpu::BindingResource::TextureView(&spot_atlas_view)),
            (6, spot_shadow_buffer.as_entire_binding()),
            (7, cluster_uniform_buffer.as_entire_binding()),
            (8, light_grid_buffer.as_entire_binding()),
            (9, light_indices_buffer.as_entire_binding()),
            (10, wgpu::BindingResource::TextureView(&albedo_array_view)),
            (11, wgpu::BindingResource::Sampler(&texture_sampler)),
            (12, wgpu::BindingResource::TextureView(&irradiance_view)),
            (13, wgpu::BindingResource::TextureView(&prefiltered_view)),
            (14, wgpu::BindingResource::TextureView(&brdf_view)),
            (15, wgpu::BindingResource::Sampler(&ibl_sampler)),
            (16, wgpu::BindingResource::TextureView(&normal_array_view)),
            (17, wgpu::BindingResource::TextureView(&orm_array_view)),
            (18, wgpu::BindingResource::TextureView(&emissive_array_view)),
            (19, wgpu::BindingResource::TextureView(&point_cube_view)),
        ],
    );
    let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SKY_SHADER.into()),
    });
    let sky_pipeline = render_pipeline(
        &device,
        &sky_shader,
        &[],
        &[Some(SCENE_FORMAT.into()), Some(SCENE_FORMAT.into())],
        false,
        wgpu::CompareFunction::Always,
    );
    let sky_bind_group = bind_group(
        &device,
        &sky_pipeline.get_bind_group_layout(0),
        vec![
            (0, camera_buffer.as_entire_binding()),
            (1, wgpu::BindingResource::TextureView(&env_cube_view)),
            (2, wgpu::BindingResource::Sampler(&ibl_sampler)),
        ],
    );
    let cluster_bounds_pipeline = compute_pipeline(&device, CLUSTER_BOUNDS_SHADER);
    let cluster_bounds_bind_group = bind_group(
        &device,
        &cluster_bounds_pipeline.get_bind_group_layout(0),
        vec![
            (0, cluster_uniform_buffer.as_entire_binding()),
            (1, cluster_bounds_buffer.as_entire_binding()),
        ],
    );
    let cluster_assign_pipeline = compute_pipeline(&device, CLUSTER_ASSIGN_SHADER);
    let cluster_assign_bind_group = bind_group(
        &device,
        &cluster_assign_pipeline.get_bind_group_layout(0),
        vec![
            (0, cluster_uniform_buffer.as_entire_binding()),
            (1, cluster_bounds_buffer.as_entire_binding()),
            (2, light_grid_buffer.as_entire_binding()),
            (3, light_indices_buffer.as_entire_binding()),
            (4, cluster_lights_buffer.as_entire_binding()),
        ],
    );
    let cascade_buffers: [wgpu::Buffer; 4] = std::array::from_fn(|_| uniform_buffer(&device, 64));
    let cascade_bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|index| {
        bind_group(
            &device,
            &shadow_pipeline.get_bind_group_layout(0),
            vec![(0, cascade_buffers[index].as_entire_binding())],
        )
    });
    let spot_buffers: [wgpu::Buffer; 4] = std::array::from_fn(|_| uniform_buffer(&device, 64));
    let spot_bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|index| {
        bind_group(
            &device,
            &shadow_pipeline.get_bind_group_layout(0),
            vec![(0, spot_buffers[index].as_entire_binding())],
        )
    });

    let ssao_pipeline = fullscreen_pipeline(&device, SSAO_SHADER, SCENE_FORMAT);
    let ssao_bind_group_layout = ssao_pipeline.get_bind_group_layout(0);
    let ssao_data_buffer = uniform_buffer(&device, 80);

    let ssao_blur_pipeline = fullscreen_pipeline(&device, SSAO_BLUR_SHADER, SCENE_FORMAT);
    let ssao_blur_bind_group_layout = ssao_blur_pipeline.get_bind_group_layout(0);

    let egui_renderer = egui_wgpu::Renderer::new(
        &device,
        surface_config.format,
        egui_wgpu::RendererOptions {
            msaa_samples: 1,
            ..Default::default()
        },
    );
    let egui_context = egui::Context::default();
    let egui_state = egui_winit::State::new(
        egui_context,
        egui::ViewportId::ROOT,
        &window,
        None,
        None,
        None,
    );

    let mut graph = RenderGraph::default();
    let background = wgpu::Color {
        r: 0.02,
        g: 0.02,
        b: 0.05,
        a: 1.0,
    };
    let swapchain = add_color_resource(&mut graph, true, surface_config.format, background);
    let scene = add_color_resource(&mut graph, false, SCENE_FORMAT, background);
    let depth = add_depth_resource(&mut graph, DEPTH_FORMAT, 0.0);
    let normals = add_color_resource(&mut graph, false, SCENE_FORMAT, wgpu::Color::BLACK);
    let ao_raw = add_color_resource(&mut graph, false, SCENE_FORMAT, wgpu::Color::BLACK);
    let ao = add_color_resource(&mut graph, false, SCENE_FORMAT, wgpu::Color::BLACK);
    let bright = add_color_resource(&mut graph, false, SCENE_FORMAT, wgpu::Color::BLACK);
    let blur_temp = add_color_resource(&mut graph, false, SCENE_FORMAT, wgpu::Color::BLACK);
    let bloom = add_color_resource(&mut graph, false, SCENE_FORMAT, wgpu::Color::BLACK);
    let ssr_raw = add_color_resource(&mut graph, false, SCENE_FORMAT, wgpu::Color::BLACK);
    let ssr = add_color_resource(&mut graph, false, SCENE_FORMAT, wgpu::Color::BLACK);
    let ldr = add_color_resource(&mut graph, false, SCENE_FORMAT, background);
    let skin_pipeline = compute_pipeline(&device, SKIN_COMPUTE_SHADER);
    let cull_pipeline = compute_pipeline(&device, MESH_CULL_SHADER);
    let joint_buffer = storage_buffer(&device, 64);
    add_pass(
        &mut graph,
        Box::new(GeometryPass {
            pipeline,
            shadow_pipeline,
            shadow_view,
            cascade_buffers,
            cascade_bind_groups,
            shadow_buffer,
            spot_atlas_view,
            spot_buffers,
            spot_bind_groups,
            spot_shadow_buffer,
            cluster_uniform_buffer,
            cluster_lights_buffer,
            cluster_bounds_pipeline,
            cluster_bounds_bind_group,
            cluster_assign_pipeline,
            cluster_assign_bind_group,
            sky_pipeline,
            sky_bind_group,
            point_pipeline,
            point_face_views,
            point_depth_view,
            point_face_buffers,
            point_face_bind_groups,
            meshes,
            skinned,
            skin_pipeline,
            joint_buffer,
            joint_capacity: 0,
            cull_pipeline,
            camera_buffer,
            lights_buffer,
            bind_group: geometry_bind_group,
        }),
        &[("color", scene), ("normals", normals), ("depth", depth)],
    );
    add_pass(
        &mut graph,
        Box::new(AutoExposurePass {
            pipeline: auto_exposure_pipeline,
            exposure_buffer: exposure_buffer.clone(),
        }),
        &[("scene", scene)],
    );
    add_pass(
        &mut graph,
        Box::new(FullscreenPass {
            pipeline: ssao_pipeline,
            bind_group_layout: ssao_bind_group_layout,
            output: "ao_raw",
            bindings: vec![
                Binding::Read("depth"),
                Binding::Read("normals"),
                Binding::Uniform,
            ],
            sampler: None,
            uniform: Some(ssao_data_buffer),
            storage: None,
            update: Some(ssao_update),
        }),
        &[("depth", depth), ("normals", normals), ("ao_raw", ao_raw)],
    );
    add_pass(
        &mut graph,
        Box::new(FullscreenPass {
            pipeline: ssao_blur_pipeline,
            bind_group_layout: ssao_blur_bind_group_layout,
            output: "ao",
            bindings: vec![
                Binding::Read("ao_raw"),
                Binding::Read("depth"),
                Binding::Read("normals"),
            ],
            sampler: None,
            uniform: None,
            storage: None,
            update: None,
        }),
        &[
            ("ao_raw", ao_raw),
            ("depth", depth),
            ("normals", normals),
            ("ao", ao),
        ],
    );
    add_pass(
        &mut graph,
        Box::new(FullscreenPass {
            pipeline: ssr_pipeline,
            bind_group_layout: ssr_bind_group_layout,
            output: "ssr_raw",
            bindings: vec![
                Binding::Read("depth"),
                Binding::Read("normals"),
                Binding::Read("scene"),
                Binding::Sampler,
                Binding::Uniform,
            ],
            sampler: Some(linear_sampler(&device)),
            uniform: Some(ssr_params_buffer),
            storage: None,
            update: Some(ssr_update),
        }),
        &[
            ("depth", depth),
            ("normals", normals),
            ("scene", scene),
            ("ssr_raw", ssr_raw),
        ],
    );
    add_pass(
        &mut graph,
        Box::new(FullscreenPass {
            pipeline: ssr_blur_pipeline,
            bind_group_layout: ssr_blur_bind_group_layout,
            output: "ssr",
            bindings: vec![
                Binding::Read("ssr_raw"),
                Binding::Read("depth"),
                Binding::Read("normals"),
                Binding::Sampler,
                Binding::Uniform,
            ],
            sampler: Some(linear_sampler(&device)),
            uniform: Some(ssr_blur_params_buffer),
            storage: None,
            update: Some(ssr_blur_update),
        }),
        &[
            ("ssr_raw", ssr_raw),
            ("depth", depth),
            ("normals", normals),
            ("ssr", ssr),
        ],
    );
    add_pass(
        &mut graph,
        Box::new(FullscreenPass {
            pipeline: bright_pipeline,
            bind_group_layout: bright_bind_group_layout,
            output: "bright",
            bindings: vec![Binding::Read("scene"), Binding::Sampler],
            sampler: Some(linear_sampler(&device)),
            uniform: None,
            storage: None,
            update: None,
        }),
        &[("scene", scene), ("bright", bright)],
    );
    add_pass(
        &mut graph,
        Box::new(FullscreenPass {
            pipeline: blur_horizontal_pipeline,
            bind_group_layout: blur_horizontal_bind_group_layout,
            output: "output",
            bindings: vec![Binding::Read("input"), Binding::Sampler, Binding::Uniform],
            sampler: Some(linear_sampler(&device)),
            uniform: Some(axis_buffer(&device, &queue, [1.0, 0.0, 0.0, 0.0])),
            storage: None,
            update: None,
        }),
        &[("input", bright), ("output", blur_temp)],
    );
    add_pass(
        &mut graph,
        Box::new(FullscreenPass {
            pipeline: blur_vertical_pipeline,
            bind_group_layout: blur_vertical_bind_group_layout,
            output: "output",
            bindings: vec![Binding::Read("input"), Binding::Sampler, Binding::Uniform],
            sampler: Some(linear_sampler(&device)),
            uniform: Some(axis_buffer(&device, &queue, [0.0, 1.0, 0.0, 0.0])),
            storage: None,
            update: None,
        }),
        &[("input", blur_temp), ("output", bloom)],
    );
    add_pass(
        &mut graph,
        Box::new(FullscreenPass {
            pipeline: composite_pipeline,
            bind_group_layout: composite_bind_group_layout,
            output: "color",
            bindings: vec![
                Binding::Read("scene"),
                Binding::Sampler,
                Binding::Read("bloom"),
                Binding::Read("ao"),
                Binding::Uniform,
                Binding::Storage,
                Binding::Read("ssr"),
            ],
            sampler: Some(linear_sampler(&device)),
            uniform: Some(composite_params_buffer),
            storage: Some(exposure_buffer.clone()),
            update: Some(composite_update),
        }),
        &[
            ("scene", scene),
            ("bloom", bloom),
            ("ao", ao),
            ("color", ldr),
            ("ssr", ssr),
        ],
    );
    add_pass(
        &mut graph,
        Box::new(FullscreenPass {
            pipeline: fxaa_pipeline,
            bind_group_layout: fxaa_bind_group_layout,
            output: "color",
            bindings: vec![Binding::Read("input"), Binding::Sampler, Binding::Uniform],
            sampler: Some(linear_sampler(&device)),
            uniform: Some(fxaa_params_buffer),
            storage: None,
            update: Some(fxaa_update),
        }),
        &[("input", ldr), ("color", swapchain)],
    );
    render_graph_compile(&mut graph);

    Graphics {
        window,
        surface,
        device,
        queue,
        surface_config,
        egui_renderer,
        egui_state,
        graph,
        size: (width, height),
        gltf: gltf_scene,
        gltf_skinned: character_scene,
    }
}

fn resize(graphics: &mut Graphics, width: u32, height: u32) {
    graphics.size = (width, height);
    graphics.surface_config.width = width;
    graphics.surface_config.height = height;
    graphics
        .surface
        .configure(&graphics.device, &graphics.surface_config);
}

fn render(graphics: &mut Graphics, world: &mut World) {
    let delta_time = world.resources.delta_time;

    #[cfg(target_arch = "wasm32")]
    {
        let canvas = get_canvas();
        if canvas.width() > 0
            && canvas.height() > 0
            && (canvas.width(), canvas.height()) != graphics.size
        {
            resize(graphics, canvas.width(), canvas.height());
        }
    }

    let (width, height) = graphics.size;
    let aspect_ratio = width as f32 / height.max(1) as f32;

    #[cfg(not(target_arch = "wasm32"))]
    let egui_input = graphics.egui_state.take_egui_input(&graphics.window);
    #[cfg(target_arch = "wasm32")]
    let mut egui_input = graphics.egui_state.take_egui_input(&graphics.window);
    #[cfg(target_arch = "wasm32")]
    {
        let pixels_per_point = graphics.egui_state.egui_ctx().pixels_per_point();
        egui_input.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(
                width as f32 / pixels_per_point,
                height as f32 / pixels_per_point,
            ),
        ));
    }

    let logical_targets = graphics
        .graph
        .resource_physical
        .iter()
        .filter(|physical| physical.is_some())
        .count();
    let physical_targets = graphics.graph.physical_formats.len();

    let mut bloom_enabled = world.resources.bloom_enabled;
    let mut ssr_enabled = world.resources.ssr_enabled;
    let egui_output = graphics.egui_state.egui_ctx().run_ui(egui_input, |ui| {
        egui::Window::new("engine").show(ui.ctx(), |ui| {
            ui.label("Spinning triangles + emissive cube");
            ui.label(format!("{:.0} fps", 1. / delta_time.max(1e-6)));
            ui.label(format!(
                "render targets: {logical_targets} logical / {physical_targets} physical"
            ));
            ui.checkbox(&mut bloom_enabled, "bloom");
            ui.checkbox(&mut ssr_enabled, "ssr");
            ui.label("drag-left orbit, drag-right pan, scroll zoom");
        });
    });
    world.resources.bloom_enabled = bloom_enabled;
    world.resources.ssr_enabled = ssr_enabled;
    graphics
        .egui_state
        .handle_platform_output(&graphics.window, egui_output.platform_output);

    let paint_jobs = graphics
        .egui_state
        .egui_ctx()
        .tessellate(egui_output.shapes, egui_output.pixels_per_point);
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [width, height],
        pixels_per_point: egui_output.pixels_per_point,
    };

    for (id, delta) in &egui_output.textures_delta.set {
        graphics
            .egui_renderer
            .update_texture(&graphics.device, &graphics.queue, *id, delta);
    }
    for id in &egui_output.textures_delta.free {
        graphics.egui_renderer.free_texture(id);
    }

    let frame = loop {
        match graphics.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => break frame,
            wgpu::CurrentSurfaceTexture::Outdated => graphics
                .surface
                .configure(&graphics.device, &graphics.surface_config),
            other => panic!("{other:?}"),
        }
    };
    let frame_view = frame.texture.create_view(&Default::default());

    let graph_commands = render_graph_execute(
        &mut graphics.graph,
        &graphics.device,
        &graphics.queue,
        world,
        aspect_ratio,
        &frame_view,
        (width, height),
    );

    let mut egui_encoder = graphics.device.create_command_encoder(&Default::default());
    graphics.egui_renderer.update_buffers(
        &graphics.device,
        &graphics.queue,
        &mut egui_encoder,
        &paint_jobs,
        &screen_descriptor,
    );
    {
        let pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        graphics
            .egui_renderer
            .render(&mut pass.forget_lifetime(), &paint_jobs, &screen_descriptor);
    }

    graphics
        .queue
        .submit([graph_commands, egui_encoder.finish()]);
    frame.present();
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.graphics.is_some() {
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        let window_attrs = Window::default_attributes();
        #[cfg(target_arch = "wasm32")]
        let (window_attrs, canvas_width, canvas_height) = {
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = get_canvas();
            let (width, height) = (canvas.width(), canvas.height());
            (
                Window::default_attributes().with_canvas(Some(canvas)),
                width,
                height,
            )
        };

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = env_logger::try_init();
            let size = window.inner_size();
            self.graphics = Some(pollster::block_on(init_graphics(
                window,
                size.width,
                size.height,
            )));
            self.world = new_world();
            self.state.initialize(&mut self.world);
            if let Some(scene) = self
                .graphics
                .as_ref()
                .and_then(|graphics| graphics.gltf.as_ref())
            {
                spawn_gltf(
                    &mut self.world,
                    scene,
                    LocalTransform {
                        translation: nalgebra_glm::vec3(0.0, 1.0, 0.0),
                        scale: nalgebra_glm::vec3(1.0, 1.0, 1.0),
                        ..Default::default()
                    },
                );
            }
            if let Some(scene) = self
                .graphics
                .as_ref()
                .and_then(|graphics| graphics.gltf_skinned.as_ref())
            {
                spawn_gltf(
                    &mut self.world,
                    scene,
                    LocalTransform {
                        translation: nalgebra_glm::vec3(2.5, 0.0, 0.0),
                        scale: nalgebra_glm::vec3(1.0, 1.0, 1.0),
                        ..Default::default()
                    },
                );
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            console_error_panic_hook::set_once();
            let _ = console_log::init();
            let (sender, receiver) = futures::channel::oneshot::channel();
            self.pending = Some(receiver);
            wasm_bindgen_futures::spawn_local(async move {
                let _ = sender.send(init_graphics(window, canvas_width, canvas_height).await);
            });
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        #[cfg(target_arch = "wasm32")]
        if let Some(receiver) = self.pending.as_mut()
            && let Ok(Some(graphics)) = receiver.try_recv()
        {
            graphics.window.request_redraw();
            self.graphics = Some(graphics);
            self.pending = None;
            self.world = new_world();
            self.state.initialize(&mut self.world);
            if let Some(scene) = self
                .graphics
                .as_ref()
                .and_then(|graphics| graphics.gltf.as_ref())
            {
                spawn_gltf(
                    &mut self.world,
                    scene,
                    LocalTransform {
                        translation: nalgebra_glm::vec3(0.0, 1.0, 0.0),
                        scale: nalgebra_glm::vec3(1.0, 1.0, 1.0),
                        ..Default::default()
                    },
                );
            }
            if let Some(scene) = self
                .graphics
                .as_ref()
                .and_then(|graphics| graphics.gltf_skinned.as_ref())
            {
                spawn_gltf(
                    &mut self.world,
                    scene,
                    LocalTransform {
                        translation: nalgebra_glm::vec3(2.5, 0.0, 0.0),
                        scale: nalgebra_glm::vec3(1.0, 1.0, 1.0),
                        ..Default::default()
                    },
                );
            }
        }

        let Some(graphics) = self.graphics.as_mut() else {
            return;
        };

        if graphics
            .egui_state
            .on_window_event(&graphics.window, &event)
            .consumed
        {
            graphics.window.request_redraw();
            return;
        }

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                resize(graphics, size.width, size.height);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                self.world
                    .resources
                    .events
                    .push(InputEvent::Button(button, pressed));
                graphics.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cursor = Vec2::new(position.x as f32, position.y as f32);
                self.world.resources.events.push(InputEvent::Cursor(cursor));
                graphics.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, vertical) => vertical,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 120.0,
                };
                self.world.resources.events.push(InputEvent::Wheel(amount));
                graphics.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                self.world.resources.viewport = (graphics.size.0 as f32, graphics.size.1 as f32);
                timing_system(&mut self.world);
                self.state.run_systems(&mut self.world);
                run_frame_systems(&mut self.world);
                render(graphics, &mut self.world);
                self.world.resources.input.position_delta = Vec2::zeros();
                self.world.resources.input.wheel_delta = 0.0;
                graphics.window.request_redraw();
            }
            _ => {}
        }
    }
}
