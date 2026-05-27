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
struct Mesh;

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

const LOCAL_TRANSFORM: u64 = 1;
const GLOBAL_TRANSFORM: u64 = 2;
const PARENT: u64 = 4;
const LOCAL_TRANSFORM_DIRTY: u64 = 8;
const CAMERA: u64 = 16;
const PAN_ORBIT_CAMERA: u64 = 32;
const MESH: u64 = 64;

macro_rules! components {
    ($($name:ty, $field:ident, $bit:expr);* $(;)?) => {
        const COMPONENT_COUNT: usize = [$($bit),*].len();

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
            $(if table.mask & $bit != 0 { table.$field.push((Default::default(), tick)); })*
            row
        }

        fn table_swap_remove(table: &mut Table, row: usize) -> Option<Entity> {
            let last = table.entities.len().saturating_sub(1);
            let moved = if row < last { Some(table.entities[last]) } else { None };
            table.entities.swap_remove(row);
            $(if table.mask & $bit != 0 { table.$field.swap_remove(row); })*
            moved
        }

        fn table_move_row(source: &mut Table, row: usize, destination: &mut Table, tick: u32) -> (usize, Option<Entity>) {
            destination.entities.push(source.entities[row]);
            $(if destination.mask & $bit != 0 {
                destination.$field.push(
                    if source.mask & $bit != 0 { std::mem::take(&mut source.$field[row]) }
                    else { (Default::default(), tick) }
                );
            })*
            let destination_row = destination.entities.len() - 1;
            (destination_row, table_swap_remove(source, row))
        }
    }
}

components!(
    LocalTransform,      local_transforms,      LOCAL_TRANSFORM;
    GlobalTransform,     global_transforms,     GLOBAL_TRANSFORM;
    Parent,              parents,               PARENT;
    LocalTransformDirty, local_transform_dirty, LOCAL_TRANSFORM_DIRTY;
    Camera,              cameras,               CAMERA;
    PanOrbitCamera,      pan_orbit_cameras,     PAN_ORBIT_CAMERA;
    Mesh,                meshes,                MESH;
);

type System = fn(&mut World);

#[derive(Default)]
struct Schedule {
    systems: Vec<(&'static str, System)>,
}

fn schedule_push(schedule: &mut Schedule, name: &'static str, system: System) {
    schedule.systems.push((name, system));
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

#[derive(Default)]
struct World {
    tables: Vec<Table>,
    table_map: HashMap<u64, usize>,
    query_cache: HashMap<u64, Vec<usize>>,
    locations: Vec<Option<Location>>,
    generations: Vec<u32>,
    free_ids: Vec<u32>,
    current_tick: u32,
    delta_time: f32,
    viewport: (f32, f32),
    active_camera: Option<Entity>,
    schedule: Schedule,
    timing: Timing,
    transform_state: TransformState,
    input: Input,
}

fn new_world() -> World {
    World {
        current_tick: 1,
        viewport: (1.0, 1.0),
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

fn get_local_transform(world: &World, entity: Entity) -> Option<LocalTransform> {
    let location = get_location(world, entity)?;
    let table = &world.tables[location.table];
    (table.mask & LOCAL_TRANSFORM != 0).then(|| table.local_transforms[location.row].0)
}

fn get_local_transform_mut(world: &mut World, entity: Entity) -> Option<&mut LocalTransform> {
    let location = get_location(world, entity)?;
    let tick = world.current_tick;
    let table = &mut world.tables[location.table];
    (table.mask & LOCAL_TRANSFORM != 0).then(|| {
        table.local_transforms[location.row].1 = tick;
        &mut table.local_transforms[location.row].0
    })
}

fn get_global_transform(world: &World, entity: Entity) -> Option<GlobalTransform> {
    let location = get_location(world, entity)?;
    let table = &world.tables[location.table];
    (table.mask & GLOBAL_TRANSFORM != 0).then(|| table.global_transforms[location.row].0)
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
    let location = get_location(world, entity)?;
    let table = &world.tables[location.table];
    (table.mask & PARENT != 0).then(|| table.parents[location.row].0)
}

fn get_camera(world: &World, entity: Entity) -> Option<Camera> {
    let location = get_location(world, entity)?;
    let table = &world.tables[location.table];
    (table.mask & CAMERA != 0).then(|| table.cameras[location.row].0)
}

fn get_pan_orbit_camera_mut(world: &mut World, entity: Entity) -> Option<&mut PanOrbitCamera> {
    let location = get_location(world, entity)?;
    let tick = world.current_tick;
    let table = &mut world.tables[location.table];
    (table.mask & PAN_ORBIT_CAMERA != 0).then(|| {
        table.pan_orbit_cameras[location.row].1 = tick;
        &mut table.pan_orbit_cameras[location.row].0
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
    if !world.transform_state.children_cache_valid {
        rebuild_children_cache(world);
    }

    let is_parent = world.transform_state.children_cache.contains_key(&entity);
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
        world.transform_state.dirty_entities.push(current);
        if let Some(children) = world.transform_state.children_cache.get(&current) {
            stack.extend(children.iter().copied());
        }
    }
}

fn rebuild_children_cache(world: &mut World) {
    world.transform_state.children_cache.clear();

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
            .transform_state
            .children_cache
            .entry(parent)
            .or_default()
            .push(child);
    }
    for children in world.transform_state.children_cache.values_mut() {
        children.sort_unstable_by_key(|entity| (entity.index, entity.generation));
    }
    world.transform_state.children_cache_valid = true;
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
    let mut dirty = std::mem::take(&mut world.transform_state.dirty_entities);
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

const SPIN_SPEED: f32 = 0.8;

fn spin_system(world: &mut World) {
    let delta_time = world.delta_time;
    for entity in collect_entities(world, MESH) {
        if let Some(local) = get_local_transform_mut(world, entity) {
            let spin = nalgebra_glm::quat_angle_axis(SPIN_SPEED * delta_time, &Vec3::y());
            local.rotation = spin * local.rotation;
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
    let Some(camera) = world.active_camera else {
        return;
    };
    if !entity_has(world, camera, PAN_ORBIT_CAMERA) {
        return;
    }

    let viewport = Vec2::new(world.viewport.0.max(1.0), world.viewport.1.max(1.0));
    let y_fov = camera_y_fov(world, camera);
    let orbit = world.input.left_pressed;
    let pan = world.input.right_pressed;
    let position_delta = world.input.position_delta;
    let wheel_delta = world.input.wheel_delta;

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

    let delta_time = world.delta_time;
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

fn view_projection_matrix(world: &World, aspect_ratio: f32) -> Option<Mat4> {
    let camera_entity = world.active_camera?;
    let camera = get_camera(world, camera_entity)?;
    let global = get_global_transform(world, camera_entity)?.0;
    let position = transform_translation(&global);
    let target = position + transform_forward(&global);
    let up = transform_up(&global);
    let view = nalgebra_glm::look_at(&position, &target, &up);
    Some(perspective_matrix(&camera.projection, aspect_ratio) * view)
}

fn renderable_models(world: &World) -> Vec<Mat4> {
    let mut models = Vec::new();
    for table in &world.tables {
        if table.mask & (MESH | GLOBAL_TRANSFORM) != (MESH | GLOBAL_TRANSFORM) {
            continue;
        }
        for entry in &table.global_transforms {
            models.push(entry.0.0);
        }
    }
    models
}

fn timing_system(world: &mut World) {
    let now = Instant::now();
    world.delta_time = world
        .timing
        .last_frame
        .map_or(0.0, |last_frame| (now - last_frame).as_secs_f32());
    world.timing.last_frame = Some(now);
    world.current_tick = world.current_tick.wrapping_add(1);
}

fn run_frame_systems(world: &mut World) {
    let schedule = std::mem::take(&mut world.schedule);
    schedule_run(&schedule, world);
    world.schedule = schedule;
}

fn initialize_world(world: &mut World) {
    schedule_push(&mut world.schedule, "spin", spin_system);
    schedule_push(&mut world.schedule, "pan_orbit", pan_orbit_camera_system);
    schedule_push(
        &mut world.schedule,
        "transforms",
        update_global_transforms_system,
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
    world.active_camera = Some(camera);
    pan_orbit_camera_system(world);

    let count = 12;
    for index in 0..count {
        let angle = index as f32 / count as f32 * std::f32::consts::TAU;
        let triangle = spawn(world, LOCAL_TRANSFORM | GLOBAL_TRANSFORM | MESH);
        if let Some(local) = get_local_transform_mut(world, triangle) {
            local.translation = nalgebra_glm::vec3(angle.cos() * 3.0, 0.0, angle.sin() * 3.0);
            local.scale = nalgebra_glm::vec3(0.6, 0.6, 0.6);
            local.rotation = nalgebra_glm::quat_angle_axis(angle, &Vec3::y());
        }
        mark_local_transform_dirty(world, triangle);
    }
}

const TRIANGLE_VERTICES: [f32; 18] = [
    1., -1., 0., 1., 0., 0., -1., -1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
];

const TINT_COUNT: u64 = 64;

const SHADER: &str = "
@group(0) @binding(0) var<uniform> view_projection: mat4x4<f32>;
@group(1) @binding(0) var<storage, read> tints: array<vec4<f32>, 64>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
}
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) instance: u32,
}
@vertex fn vs(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let clip = view_projection * model * vec4<f32>(in.position, 1.0);
    return VertexOutput(clip, vec4<f32>(in.color, 1.0), instance_index);
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let tint = tints[in.instance % 64u].rgb;
    return vec4<f32>(in.color.rgb * tint, 1.0);
}
";

const COMPUTE_SHADER: &str = "
@group(0) @binding(0) var<storage, read_write> tints: array<vec4<f32>, 64>;
@group(0) @binding(1) var<uniform> params: vec4<f32>;
@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= 64u) {
        return;
    }
    let phase = params.x + f32(index) * 0.3;
    tints[index] = vec4<f32>(
        0.5 + 0.5 * sin(phase),
        0.5 + 0.5 * sin(phase + 2.094),
        0.5 + 0.5 * sin(phase + 4.188),
        1.0,
    );
}
";

const TONEMAP_SHADER: &str = "
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32((vertex_index & 1u) << 1u);
    let y = f32(vertex_index & 2u);
    var out: VertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}
@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
fn aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(scene_texture, scene_sampler, in.uv).rgb;
    return vec4<f32>(aces(color), 1.0);
}
";

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone, Copy)]
enum Clear {
    Color(wgpu::Color),
    Depth(f32),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResourceKind {
    Texture,
    Buffer,
}

struct ResourceDesc {
    kind: ResourceKind,
    external: bool,
    format: wgpu::TextureFormat,
    clear: Clear,
    buffer_size: u64,
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
    buffers: Vec<Option<wgpu::Buffer>>,
    size: (u32, u32),
}

struct PassContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    encoder: &'a mut wgpu::CommandEncoder,
    world: &'a World,
    aspect_ratio: f32,
    resources: &'a [ResourceDesc],
    views: &'a [Option<&'a wgpu::TextureView>],
    buffers: &'a [Option<&'a wgpu::Buffer>],
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
    fn buffer_writes(&self) -> Vec<&'static str> {
        Vec::new()
    }
    fn execute(&mut self, context: &mut PassContext);
}

fn write_slots(node: &dyn PassNode) -> Vec<&'static str> {
    let mut slots = node.color_writes();
    slots.extend(node.depth_write());
    slots.extend(node.buffer_writes());
    slots
}

fn add_color_resource(
    graph: &mut RenderGraph,
    external: bool,
    format: wgpu::TextureFormat,
    clear_color: wgpu::Color,
) -> usize {
    graph.resources.push(ResourceDesc {
        kind: ResourceKind::Texture,
        external,
        format,
        clear: Clear::Color(clear_color),
        buffer_size: 0,
    });
    graph.resources.len() - 1
}

fn add_depth_resource(
    graph: &mut RenderGraph,
    format: wgpu::TextureFormat,
    clear_depth: f32,
) -> usize {
    graph.resources.push(ResourceDesc {
        kind: ResourceKind::Texture,
        external: false,
        format,
        clear: Clear::Depth(clear_depth),
        buffer_size: 0,
    });
    graph.resources.len() - 1
}

fn add_buffer_resource(graph: &mut RenderGraph, size: u64) -> usize {
    graph.resources.push(ResourceDesc {
        kind: ResourceKind::Buffer,
        external: false,
        format: SCENE_FORMAT,
        clear: Clear::Depth(0.0),
        buffer_size: size,
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
        .filter(|&resource| {
            used[resource]
                && !graph.resources[resource].external
                && graph.resources[resource].kind == ResourceKind::Texture
        })
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
    graph.buffers = (0..resource_count).map(|_| None).collect();
    graph.size = (0, 0);
    graph.execution_order = order;
}

fn ensure_resources(graph: &mut RenderGraph, device: &wgpu::Device, size: (u32, u32)) {
    for resource in 0..graph.resources.len() {
        if graph.resources[resource].kind == ResourceKind::Buffer
            && graph.buffers[resource].is_none()
        {
            graph.buffers[resource] = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: graph.resources[resource].buffer_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }
    }

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

fn graph_buffer<'a>(context: &PassContext<'a>, slot: &str) -> &'a wgpu::Buffer {
    let resource = context.bindings[slot];
    context.buffers[resource].expect("unbound buffer resource")
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
    let buffers: Vec<Option<&wgpu::Buffer>> =
        graph.buffers.iter().map(|buffer| buffer.as_ref()).collect();

    let mut encoder = device.create_command_encoder(&Default::default());
    for index in graph.execution_order.clone() {
        let bindings = graph.passes[index].bindings.clone();
        let mut context = PassContext {
            device,
            queue,
            encoder: &mut encoder,
            world,
            aspect_ratio,
            resources: &graph.resources,
            views: &views,
            buffers: &buffers,
            bindings: &bindings,
            clears: &graph.clears,
            stores: &graph.stores,
            pass_index: index,
        };
        graph.passes[index].node.execute(&mut context);
    }
    encoder.finish()
}

struct TrianglePass {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    view_projection_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    tint_bind_group_layout: wgpu::BindGroupLayout,
}

impl PassNode for TrianglePass {
    fn reads(&self) -> Vec<&'static str> {
        vec!["tints"]
    }

    fn color_writes(&self) -> Vec<&'static str> {
        vec!["color"]
    }

    fn depth_write(&self) -> Option<&'static str> {
        Some("depth")
    }

    fn execute(&mut self, context: &mut PassContext) {
        let view_projection = view_projection_matrix(context.world, context.aspect_ratio)
            .unwrap_or_else(Mat4::identity);
        context.queue.write_buffer(
            &self.view_projection_buffer,
            0,
            bytemuck::cast_slice(view_projection.as_slice()),
        );

        let models = renderable_models(context.world);
        let instance_count = models.len() as u32;
        if instance_count > self.instance_capacity {
            self.instance_capacity = instance_count.next_power_of_two();
            self.instance_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: self.instance_capacity as u64 * 64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if instance_count > 0 {
            let mut instance_data: Vec<f32> = Vec::with_capacity(models.len() * 16);
            for model in &models {
                instance_data.extend_from_slice(model.as_slice());
            }
            context.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instance_data),
            );
        }

        let tints = graph_buffer(context, "tints");
        let tint_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.tint_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tints.as_entire_binding(),
                }],
            });

        let (color_view, color_load, color_store) = color_attachment(context, "color");
        let (depth_view, depth_load, depth_store) = depth_attachment(context, "depth");
        let mut pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: color_load,
                        store: color_store,
                    },
                })],
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
        if instance_count > 0 {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &tint_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            pass.draw(0..3, 0..instance_count);
        }
    }
}

struct TonemapPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl PassNode for TonemapPass {
    fn reads(&self) -> Vec<&'static str> {
        vec!["scene"]
    }

    fn color_writes(&self) -> Vec<&'static str> {
        vec!["color"]
    }

    fn execute(&mut self, context: &mut PassContext) {
        let scene_view = read_view(context, "scene");
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

        let (color_view, color_load, color_store) = color_attachment(context, "color");
        let mut pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: color_load,
                        store: color_store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

struct ComputeTintPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    time: f32,
}

impl PassNode for ComputeTintPass {
    fn buffer_writes(&self) -> Vec<&'static str> {
        vec!["tints"]
    }

    fn execute(&mut self, context: &mut PassContext) {
        self.time += context.world.delta_time;
        let params = [self.time, 0.0, 0.0, 0.0];
        context
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        let tints = graph_buffer(context, "tints");
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tints.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut pass = context
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
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
}

#[derive(Default)]
pub struct App {
    world: World,
    graphics: Option<Graphics>,
    #[cfg(target_arch = "wasm32")]
    pending: Option<futures::channel::oneshot::Receiver<Graphics>>,
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
    let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
    let instance_attrs =
        wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attrs,
                },
                wgpu::VertexBufferLayout {
                    array_stride: 64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &instance_attrs,
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(SCENE_FORMAT.into())],
        }),
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let tonemap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(TONEMAP_SHADER.into()),
    });
    let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &tonemap_shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &tonemap_shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(surface_config.format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let tonemap_bind_group_layout = tonemap_pipeline.get_bind_group_layout(0);
    let tonemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of_val(&TRIANGLE_VERTICES) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&TRIANGLE_VERTICES));

    let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let view_projection_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: view_projection_buffer.as_entire_binding(),
        }],
    });
    let tint_bind_group_layout = pipeline.get_bind_group_layout(1);

    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &compute_shader,
        entry_point: Some("cs"),
        compilation_options: Default::default(),
        cache: None,
    });
    let compute_bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let compute_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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
    let tints = add_buffer_resource(&mut graph, TINT_COUNT * 16);
    add_pass(
        &mut graph,
        Box::new(ComputeTintPass {
            pipeline: compute_pipeline,
            bind_group_layout: compute_bind_group_layout,
            params_buffer: compute_params_buffer,
            time: 0.0,
        }),
        &[("tints", tints)],
    );
    add_pass(
        &mut graph,
        Box::new(TrianglePass {
            pipeline,
            vertex_buffer,
            instance_buffer,
            instance_capacity: 1,
            view_projection_buffer,
            bind_group,
            tint_bind_group_layout,
        }),
        &[("color", scene), ("depth", depth), ("tints", tints)],
    );
    add_pass(
        &mut graph,
        Box::new(TonemapPass {
            pipeline: tonemap_pipeline,
            bind_group_layout: tonemap_bind_group_layout,
            sampler: tonemap_sampler,
        }),
        &[("scene", scene), ("color", swapchain)],
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

fn render(graphics: &mut Graphics, world: &World) {
    let delta_time = world.delta_time;

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

    let egui_output = graphics.egui_state.egui_ctx().run_ui(egui_input, |ui| {
        egui::Window::new("engine").show(ui.ctx(), |ui| {
            ui.label("Spinning triangles");
            ui.label(format!("{:.0} fps", 1. / delta_time.max(1e-6)));
            ui.label("drag-left orbit, drag-right pan, scroll zoom");
        });
    });
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
            initialize_world(&mut self.world);
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
            initialize_world(&mut self.world);
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
                match button {
                    MouseButton::Left => self.world.input.left_pressed = pressed,
                    MouseButton::Right => self.world.input.right_pressed = pressed,
                    _ => {}
                }
                graphics.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cursor = Vec2::new(position.x as f32, position.y as f32);
                if self.world.input.cursor_initialized {
                    self.world.input.position_delta += cursor - self.world.input.cursor;
                } else {
                    self.world.input.cursor_initialized = true;
                }
                self.world.input.cursor = cursor;
                graphics.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.world.input.wheel_delta += match delta {
                    MouseScrollDelta::LineDelta(_, vertical) => vertical,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 120.0,
                };
                graphics.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                self.world.viewport = (graphics.size.0 as f32, graphics.size.1 as f32);
                timing_system(&mut self.world);
                run_frame_systems(&mut self.world);
                render(graphics, &self.world);
                self.world.input.position_delta = Vec2::zeros();
                self.world.input.wheel_delta = 0.0;
                graphics.window.request_redraw();
            }
            _ => {}
        }
    }
}
