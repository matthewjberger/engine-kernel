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

#[derive(Clone, Copy, Default)]
struct Light {
    color: Vec3,
    intensity: f32,
    point: bool,
    range: f32,
}

#[derive(Clone, Copy)]
struct Material {
    albedo: Vec3,
    metallic: f32,
    roughness: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            metallic: 0.0,
            roughness: 0.6,
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

macro_rules! components {
    (@bits $index:expr,) => {};
    (@bits $index:expr, $constant:ident $($rest:ident)*) => {
        const $constant: u64 = 1u64 << $index;
        components!(@bits $index + 1, $($rest)*);
    };
    ($($constant:ident => $name:ty, $field:ident);* $(;)?) => {
        components!(@bits 0u64, $($constant)*);

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

components!(
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

fn gather_lights(world: &World) -> [f32; 80] {
    let mut data = [0.0f32; 80];
    data[0] = 0.03;
    data[1] = 0.03;
    data[2] = 0.04;
    let mut point_count = 0usize;
    for table in &world.tables {
        if table.mask & (LIGHT | GLOBAL_TRANSFORM) != (LIGHT | GLOBAL_TRANSFORM) {
            continue;
        }
        for row in 0..table.entities.len() {
            let global = table.global_transforms[row].0.0;
            let light = table.lights[row].0;
            let color = light.color * light.intensity;
            if light.point {
                if point_count < 8 {
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
                    point_count += 1;
                }
            } else {
                let direction = -transform_forward(&global);
                data[4] = direction.x;
                data[5] = direction.y;
                data[6] = direction.z;
                data[7] = 1.0;
                data[8] = color.x;
                data[9] = color.y;
                data[10] = color.z;
            }
        }
    }
    data[12] = point_count as f32;
    data
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
            light.point = true;
            light.range = 7.0;
        }
        mark_local_transform_dirty(world, lamp);

        let count = 12;
        for index in 0..count {
            let angle = index as f32 / count as f32 * std::f32::consts::TAU;
            world.resources.commands.push(SpawnCommand {
                mask: LOCAL_TRANSFORM | GLOBAL_TRANSFORM | RENDER_MESH | MATERIAL,
                transform: LocalTransform {
                    translation: nalgebra_glm::vec3(angle.cos() * 3.0, 0.0, angle.sin() * 3.0),
                    rotation: nalgebra_glm::quat_angle_axis(angle, &Vec3::y()),
                    scale: nalgebra_glm::vec3(0.6, 0.6, 0.6),
                },
                render_mesh: 0,
                emissive: Vec3::zeros(),
                material: Material {
                    albedo: nalgebra_glm::vec3(1.0, 1.0, 1.0),
                    metallic: 0.85,
                    roughness: 0.35,
                },
            });
        }

        world.resources.commands.push(SpawnCommand {
            mask: LOCAL_TRANSFORM | GLOBAL_TRANSFORM | RENDER_MESH | EMISSIVE,
            transform: LocalTransform {
                scale: nalgebra_glm::vec3(0.8, 0.8, 0.8),
                ..Default::default()
            },
            render_mesh: 1,
            emissive: nalgebra_glm::vec3(4.0, 2.2, 0.8),
            material: Material::default(),
        });

        world.resources.commands.push(SpawnCommand {
            mask: LOCAL_TRANSFORM | GLOBAL_TRANSFORM | RENDER_MESH | MATERIAL,
            transform: LocalTransform {
                translation: nalgebra_glm::vec3(0.0, -1.0, 0.0),
                scale: nalgebra_glm::vec3(10.0, 0.1, 10.0),
                ..Default::default()
            },
            render_mesh: 1,
            emissive: Vec3::zeros(),
            material: Material {
                albedo: nalgebra_glm::vec3(0.7, 0.7, 0.7),
                metallic: 0.0,
                roughness: 0.9,
            },
        });
    }
}

const TRIANGLE_VERTICES: [f32; 27] = [
    1., -1., 0., 1., 0., 0., 0., 0., 1., -1., -1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
    1., 0., 0., 1.,
];

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

const SHADER: &str = "
struct Camera {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec4<f32>,
}
struct Lights {
    ambient: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    point_count: vec4<f32>,
    point_position: array<vec4<f32>, 8>,
    point_color: array<vec4<f32>, 8>,
}
struct Shadow {
    cascade_view_projection: array<mat4x4<f32>, 4>,
    split_distances: vec4<f32>,
    atlas_offset: array<vec4<f32>, 4>,
    atlas_scale: vec4<f32>,
    params: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lights: Lights;
@group(0) @binding(2) var shadow_texture: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler;
@group(0) @binding(4) var<uniform> shadow: Shadow;

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
) -> vec3<f32> {
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    var color = lights.ambient.rgb * albedo;
    let sun = lights.sun_direction.xyz;
    let view_depth = -(camera.view * vec4<f32>(world_position, 1.0)).z;
    let sun_shadow = calculate_shadow(world_position, normal, view_depth);
    color += brdf(normal, view, sun, lights.sun_color.rgb * sun_shadow, albedo, metallic, roughness, f0);
    let count = u32(lights.point_count.x);
    for (var index = 0u; index < count; index = index + 1u) {
        let to_light = lights.point_position[index].xyz - world_position;
        let distance = max(length(to_light), 0.0001);
        let attenuation = range_attenuation(lights.point_position[index].w, distance);
        let radiance = lights.point_color[index].rgb * attenuation;
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
    return VertexOutput(
        clip,
        vec4<f32>(in.color, 1.0),
        in.emissive.rgb,
        view_normal,
        world.xyz,
        world_normal,
        in.albedo_metallic.rgb,
        vec2<f32>(in.albedo_metallic.w, in.emissive.w),
    );
}
@fragment fn fs(in: VertexOutput) -> GeometryOutput {
    let emissive_surface = in.emissive.r + in.emissive.g + in.emissive.b > 0.0;
    let normal = normalize(in.world_normal);
    let view = normalize(camera.camera_position.xyz - in.world_position);
    let albedo = in.color.rgb * in.albedo;
    let lit = lighting(albedo, in.world_position, normal, view, in.material.x, max(in.material.y, 0.04));
    let shaded = select(lit, in.color.rgb * in.emissive, emissive_surface);
    var out: GeometryOutput;
    out.color = vec4<f32>(shaded, 1.0);
    out.normal = vec4<f32>(normalize(in.view_normal), 1.0);
    return out;
}
";

const SHADOW_SHADER: &str = "
@group(0) @binding(0) var<uniform> cascade_view_projection: mat4x4<f32>;
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
}
@vertex fn vs(in: VertexInput) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    return cascade_view_projection * model * vec4<f32>(in.position, 1.0);
}
";

const FULLSCREEN_VERTEX: &str = "
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
";

const BRIGHT_SHADER: &str = "
@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(scene_texture, scene_sampler, in.uv).rgb;
    return vec4<f32>(max(color - vec3<f32>(1.0), vec3<f32>(0.0)), 1.0);
}
";

const BLUR_SHADER: &str = "
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> axis: vec4<f32>;
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let step = axis.xy / vec2<f32>(textureDimensions(input_texture));
    var sum = textureSample(input_texture, input_sampler, in.uv).rgb * 0.227027;
    sum += textureSample(input_texture, input_sampler, in.uv + step * 1.0).rgb * 0.194594;
    sum += textureSample(input_texture, input_sampler, in.uv - step * 1.0).rgb * 0.194594;
    sum += textureSample(input_texture, input_sampler, in.uv + step * 2.0).rgb * 0.121622;
    sum += textureSample(input_texture, input_sampler, in.uv - step * 2.0).rgb * 0.121622;
    sum += textureSample(input_texture, input_sampler, in.uv + step * 3.0).rgb * 0.054054;
    sum += textureSample(input_texture, input_sampler, in.uv - step * 3.0).rgb * 0.054054;
    sum += textureSample(input_texture, input_sampler, in.uv + step * 4.0).rgb * 0.016216;
    sum += textureSample(input_texture, input_sampler, in.uv - step * 4.0).rgb * 0.016216;
    return vec4<f32>(sum, 1.0);
}
";

const SSAO_SHADER: &str = "
struct Ssao {
    inverse_projection: mat4x4<f32>,
    params: vec4<f32>,
}
@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> data: Ssao;
fn view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let position = data.inverse_projection * vec4<f32>(ndc, 1.0);
    return position.xyz / position.w;
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(depth_texture));
    let coord = vec2<i32>(in.uv * dimensions);
    let depth = textureLoad(depth_texture, coord, 0);
    if (depth <= 0.0) {
        return vec4<f32>(1.0);
    }
    let position = view_position(in.uv, depth);
    let normal = normalize(textureLoad(normal_texture, coord, 0).xyz);
    let radius = data.params.x;
    let bias = data.params.y;
    let strength = data.params.z;
    var occlusion = 0.0;
    for (var index = 0; index < 8; index = index + 1) {
        let angle = f32(index) / 8.0 * 6.2831853;
        let sample_coord = coord + vec2<i32>(vec2<f32>(cos(angle), sin(angle)) * radius);
        let sample_depth = textureLoad(depth_texture, sample_coord, 0);
        if (sample_depth <= 0.0) {
            continue;
        }
        let sample_uv = (vec2<f32>(sample_coord) + 0.5) / dimensions;
        let difference = view_position(sample_uv, sample_depth) - position;
        let range = 1.0 / (1.0 + dot(difference, difference));
        occlusion += max(dot(normalize(difference), normal) - bias, 0.0) * range;
    }
    let ao = clamp(1.0 - occlusion / 8.0 * strength, 0.0, 1.0);
    return vec4<f32>(vec3<f32>(ao), 1.0);
}
";

const SSAO_BLUR_SHADER: &str = "
@group(0) @binding(0) var ao_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.uv * vec2<f32>(textureDimensions(ao_texture)));
    let center_depth = textureLoad(depth_texture, coord, 0);
    if (center_depth <= 0.0) {
        return vec4<f32>(1.0);
    }
    let center_normal = normalize(textureLoad(normal_texture, coord, 0).xyz);
    var total = 0.0;
    var weight_total = 0.0;
    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let sample_coord = coord + vec2<i32>(x, y);
            let sample_depth = textureLoad(depth_texture, sample_coord, 0);
            if (sample_depth <= 0.0) {
                continue;
            }
            let sample_normal = normalize(textureLoad(normal_texture, sample_coord, 0).xyz);
            let depth_weight = exp(-abs(sample_depth - center_depth) * 200.0);
            let normal_weight = pow(max(dot(sample_normal, center_normal), 0.0), 8.0);
            let weight = depth_weight * normal_weight;
            total += textureLoad(ao_texture, sample_coord, 0).r * weight;
            weight_total += weight;
        }
    }
    return vec4<f32>(vec3<f32>(total / max(weight_total, 0.0001)), 1.0);
}
";

const COMPOSITE_SHADER: &str = "
@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(3) var ao_texture: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: vec4<f32>;
fn aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_texture, scene_sampler, in.uv).rgb;
    let bloom = textureSample(bloom_texture, scene_sampler, in.uv).rgb;
    let ao = textureSample(ao_texture, scene_sampler, in.uv).r;
    return vec4<f32>(aces(scene * ao + bloom * params.x), 1.0);
}
";

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
}

struct GeometryPass {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_view: wgpu::TextureView,
    cascade_buffers: [wgpu::Buffer; 4],
    cascade_bind_groups: [wgpu::BindGroup; 4],
    shadow_buffer: wgpu::Buffer,
    meshes: Vec<MeshGpu>,
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
            size: *capacity as u64 * 96,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
    if count > 0 {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }
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
        size: 96,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    MeshGpu {
        vertex_buffer,
        vertex_count: vertices.len() as u32 / 9,
        instance_buffer,
        instance_capacity: 1,
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

        let lights = gather_lights(context.world);
        let sun = nalgebra_glm::vec3(lights[4], lights[5], lights[6]);
        let camera_position = view
            .try_inverse()
            .map_or(Vec3::zeros(), |inverse| transform_translation(&inverse));
        let mut camera_data = [0.0f32; 36];
        camera_data[..16].copy_from_slice(view_projection.as_slice());
        camera_data[16..32].copy_from_slice(view.as_slice());
        camera_data[32..35].copy_from_slice(camera_position.as_slice());
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

        let mut counts = Vec::with_capacity(self.meshes.len());
        for (handle, mesh) in self.meshes.iter_mut().enumerate() {
            let instances = mesh_instances(context.world, handle as u32);
            let count = instances.len() as u32;
            let mut data: Vec<f32> = Vec::with_capacity(instances.len() * 24);
            for (model, emissive, material) in &instances {
                data.extend_from_slice(model.as_slice());
                data.extend_from_slice(&[emissive.x, emissive.y, emissive.z, material.roughness]);
                data.extend_from_slice(&[
                    material.albedo.x,
                    material.albedo.y,
                    material.albedo.z,
                    material.metallic,
                ]);
            }
            upload_instances(
                context.device,
                context.queue,
                &mut mesh.instance_buffer,
                &mut mesh.instance_capacity,
                &data,
                count,
            );
            counts.push(count);
        }

        for cascade in 0..4 {
            let load = if cascade == 0 {
                wgpu::LoadOp::Clear(0.0)
            } else {
                wgpu::LoadOp::Load
            };
            let mut shadow_pass = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_view,
                        depth_ops: Some(wgpu::Operations {
                            load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
            let slot_x = (cascade % 2) as f32 * 1024.0;
            let slot_y = (cascade / 2) as f32 * 1024.0;
            shadow_pass.set_viewport(slot_x, slot_y, 1024.0, 1024.0, 0.0, 1.0);
            shadow_pass.set_scissor_rect(slot_x as u32, slot_y as u32, 1024, 1024);
            shadow_pass.set_pipeline(&self.shadow_pipeline);
            shadow_pass.set_bind_group(0, &self.cascade_bind_groups[cascade], &[]);
            for (handle, mesh) in self.meshes.iter().enumerate() {
                if counts[handle] > 0 {
                    shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    shadow_pass.set_vertex_buffer(1, mesh.instance_buffer.slice(..));
                    shadow_pass.draw(0..mesh.vertex_count, 0..counts[handle]);
                }
            }
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        for (handle, mesh) in self.meshes.iter().enumerate() {
            if counts[handle] > 0 {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, mesh.instance_buffer.slice(..));
                pass.draw(0..mesh.vertex_count, 0..counts[handle]);
            }
        }
    }
}

struct BrightPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl PassNode for BrightPass {
    fn reads(&self) -> Vec<&'static str> {
        vec!["scene"]
    }

    fn color_writes(&self) -> Vec<&'static str> {
        vec!["bright"]
    }

    fn execute(&mut self, context: &mut PassContext) {
        let scene_view = read_view(context, "scene");
        let bind_group = bind_group(
            context.device,
            &self.bind_group_layout,
            vec![
                (0, wgpu::BindingResource::TextureView(scene_view)),
                (1, wgpu::BindingResource::Sampler(&self.sampler)),
            ],
        );

        fullscreen_pass(context, &self.pipeline, &bind_group, "bright");
    }
}

struct BlurPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    axis_buffer: wgpu::Buffer,
}

impl PassNode for BlurPass {
    fn reads(&self) -> Vec<&'static str> {
        vec!["input"]
    }

    fn color_writes(&self) -> Vec<&'static str> {
        vec!["output"]
    }

    fn execute(&mut self, context: &mut PassContext) {
        let input_view = read_view(context, "input");
        let bind_group = bind_group(
            context.device,
            &self.bind_group_layout,
            vec![
                (0, wgpu::BindingResource::TextureView(input_view)),
                (1, wgpu::BindingResource::Sampler(&self.sampler)),
                (2, self.axis_buffer.as_entire_binding()),
            ],
        );

        fullscreen_pass(context, &self.pipeline, &bind_group, "output");
    }
}

struct SsaoPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    data_buffer: wgpu::Buffer,
}

impl PassNode for SsaoPass {
    fn reads(&self) -> Vec<&'static str> {
        vec!["depth", "normals"]
    }

    fn color_writes(&self) -> Vec<&'static str> {
        vec!["ao_raw"]
    }

    fn execute(&mut self, context: &mut PassContext) {
        let projection =
            camera_projection(context.world, context.aspect_ratio).unwrap_or_else(Mat4::identity);
        let inverse_projection = projection.try_inverse().unwrap_or_else(Mat4::identity);
        let mut data = [0.0f32; 20];
        data[..16].copy_from_slice(inverse_projection.as_slice());
        data[16..].copy_from_slice(&[24.0, 0.025, 2.5, 0.0]);
        context
            .queue
            .write_buffer(&self.data_buffer, 0, bytemuck::cast_slice(&data));

        let depth_view = read_view(context, "depth");
        let normal_view = read_view(context, "normals");
        let bind_group = bind_group(
            context.device,
            &self.bind_group_layout,
            vec![
                (0, wgpu::BindingResource::TextureView(depth_view)),
                (1, wgpu::BindingResource::TextureView(normal_view)),
                (2, self.data_buffer.as_entire_binding()),
            ],
        );

        fullscreen_pass(context, &self.pipeline, &bind_group, "ao_raw");
    }
}

struct SsaoBlurPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl PassNode for SsaoBlurPass {
    fn reads(&self) -> Vec<&'static str> {
        vec!["ao_raw", "depth", "normals"]
    }

    fn color_writes(&self) -> Vec<&'static str> {
        vec!["ao"]
    }

    fn execute(&mut self, context: &mut PassContext) {
        let ao_view = read_view(context, "ao_raw");
        let depth_view = read_view(context, "depth");
        let normal_view = read_view(context, "normals");
        let bind_group = bind_group(
            context.device,
            &self.bind_group_layout,
            vec![
                (0, wgpu::BindingResource::TextureView(ao_view)),
                (1, wgpu::BindingResource::TextureView(depth_view)),
                (2, wgpu::BindingResource::TextureView(normal_view)),
            ],
        );

        fullscreen_pass(context, &self.pipeline, &bind_group, "ao");
    }
}

struct CompositePass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
}

impl PassNode for CompositePass {
    fn reads(&self) -> Vec<&'static str> {
        vec!["scene", "bloom", "ao"]
    }

    fn color_writes(&self) -> Vec<&'static str> {
        vec!["color"]
    }

    fn execute(&mut self, context: &mut PassContext) {
        let bloom: f32 = if context.world.resources.bloom_enabled {
            0.8
        } else {
            0.0
        };
        context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[bloom, 0.0, 0.0, 0.0]),
        );

        let scene_view = read_view(context, "scene");
        let bloom_view = read_view(context, "bloom");
        let ao_view = read_view(context, "ao");
        let bind_group = bind_group(
            context.device,
            &self.bind_group_layout,
            vec![
                (0, wgpu::BindingResource::TextureView(scene_view)),
                (1, wgpu::BindingResource::Sampler(&self.sampler)),
                (2, wgpu::BindingResource::TextureView(bloom_view)),
                (3, wgpu::BindingResource::TextureView(ao_view)),
                (4, self.params_buffer.as_entire_binding()),
            ],
        );

        fullscreen_pass(context, &self.pipeline, &bind_group, "color");
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
    let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 7 => Float32x3];
    let instance_attrs = wgpu::vertex_attr_array![
        2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 8 => Float32x4
    ];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: 36,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attrs,
                },
                wgpu::VertexBufferLayout {
                    array_stride: 96,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &instance_attrs,
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(SCENE_FORMAT.into()), Some(SCENE_FORMAT.into())],
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

    let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
    });
    let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &shadow_shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: 36,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attrs,
                },
                wgpu::VertexBufferLayout {
                    array_stride: 96,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &instance_attrs,
                },
            ],
        },
        fragment: None,
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
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
    });
    let shadow_view = shadow_texture.create_view(&Default::default());
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

    let composite_pipeline = fullscreen_pipeline(&device, COMPOSITE_SHADER, surface_config.format);
    let composite_bind_group_layout = composite_pipeline.get_bind_group_layout(0);
    let composite_params_buffer = uniform_buffer(&device, 16);

    let meshes = vec![
        mesh_gpu(&device, &queue, &TRIANGLE_VERTICES),
        mesh_gpu(&device, &queue, &CUBE_VERTICES),
    ];

    let camera_buffer = uniform_buffer(&device, 144);
    let lights_buffer = uniform_buffer(&device, 320);
    let shadow_buffer = uniform_buffer(&device, 368);
    let geometry_bind_group = bind_group(
        &device,
        &pipeline.get_bind_group_layout(0),
        vec![
            (0, camera_buffer.as_entire_binding()),
            (1, lights_buffer.as_entire_binding()),
            (2, wgpu::BindingResource::TextureView(&shadow_view)),
            (3, wgpu::BindingResource::Sampler(&shadow_sampler)),
            (4, shadow_buffer.as_entire_binding()),
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
    add_pass(
        &mut graph,
        Box::new(GeometryPass {
            pipeline,
            shadow_pipeline,
            shadow_view,
            cascade_buffers,
            cascade_bind_groups,
            shadow_buffer,
            meshes,
            camera_buffer,
            lights_buffer,
            bind_group: geometry_bind_group,
        }),
        &[("color", scene), ("normals", normals), ("depth", depth)],
    );
    add_pass(
        &mut graph,
        Box::new(SsaoPass {
            pipeline: ssao_pipeline,
            bind_group_layout: ssao_bind_group_layout,
            data_buffer: ssao_data_buffer,
        }),
        &[("depth", depth), ("normals", normals), ("ao_raw", ao_raw)],
    );
    add_pass(
        &mut graph,
        Box::new(SsaoBlurPass {
            pipeline: ssao_blur_pipeline,
            bind_group_layout: ssao_blur_bind_group_layout,
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
        Box::new(BrightPass {
            pipeline: bright_pipeline,
            bind_group_layout: bright_bind_group_layout,
            sampler: linear_sampler(&device),
        }),
        &[("scene", scene), ("bright", bright)],
    );
    add_pass(
        &mut graph,
        Box::new(BlurPass {
            pipeline: blur_horizontal_pipeline,
            bind_group_layout: blur_horizontal_bind_group_layout,
            sampler: linear_sampler(&device),
            axis_buffer: axis_buffer(&device, &queue, [1.0, 0.0, 0.0, 0.0]),
        }),
        &[("input", bright), ("output", blur_temp)],
    );
    add_pass(
        &mut graph,
        Box::new(BlurPass {
            pipeline: blur_vertical_pipeline,
            bind_group_layout: blur_vertical_bind_group_layout,
            sampler: linear_sampler(&device),
            axis_buffer: axis_buffer(&device, &queue, [0.0, 1.0, 0.0, 0.0]),
        }),
        &[("input", blur_temp), ("output", bloom)],
    );
    add_pass(
        &mut graph,
        Box::new(CompositePass {
            pipeline: composite_pipeline,
            bind_group_layout: composite_bind_group_layout,
            sampler: linear_sampler(&device),
            params_buffer: composite_params_buffer,
        }),
        &[
            ("scene", scene),
            ("bloom", bloom),
            ("ao", ao),
            ("color", swapchain),
        ],
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
    let egui_output = graphics.egui_state.egui_ctx().run_ui(egui_input, |ui| {
        egui::Window::new("engine").show(ui.ctx(), |ui| {
            ui.label("Spinning triangles + emissive cube");
            ui.label(format!("{:.0} fps", 1. / delta_time.max(1e-6)));
            ui.label(format!(
                "render targets: {logical_targets} logical / {physical_targets} physical"
            ));
            ui.checkbox(&mut bloom_enabled, "bloom");
            ui.label("drag-left orbit, drag-right pan, scroll zoom");
        });
    });
    world.resources.bloom_enabled = bloom_enabled;
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
