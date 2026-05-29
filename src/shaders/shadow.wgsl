@group(0) @binding(0) var<uniform> cascade_view_projection: mat4x4<f32>;
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
@vertex fn vs(in: VertexInput, @builtin(instance_index) instance_index: u32) -> @builtin(position) vec4<f32> {
    let object = objects[instance_index];
    let model = mat4x4<f32>(object.model_0, object.model_1, object.model_2, object.model_3);
    return cascade_view_projection * model * vec4<f32>(in.position, 1.0);
}
