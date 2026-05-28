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
