struct PointFace {
    view_projection: mat4x4<f32>,
    light: vec4<f32>,
}
@group(0) @binding(0) var<uniform> face: PointFace;
struct Instance {
    model_0: vec4<f32>,
    model_1: vec4<f32>,
    model_2: vec4<f32>,
    model_3: vec4<f32>,
    emissive: vec4<f32>,
    albedo_metallic: vec4<f32>,
    layers: vec4<f32>,
}
@group(1) @binding(0) var<storage, read> objects: array<Instance>;
struct VertexInput {
    @location(0) position: vec3<f32>,
}
struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}
@vertex fn vs(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let object = objects[instance_index];
    let model = mat4x4<f32>(object.model_0, object.model_1, object.model_2, object.model_3);
    let world = model * vec4<f32>(in.position, 1.0);
    return VertexOutput(face.view_projection * world, world.xyz);
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let distance = length(in.world_position - face.light.xyz) / face.light.w;
    return vec4<f32>(distance, 0.0, 0.0, 1.0);
}
