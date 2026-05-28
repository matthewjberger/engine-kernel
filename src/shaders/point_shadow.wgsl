struct PointFace {
    view_projection: mat4x4<f32>,
    light: vec4<f32>,
}
@group(0) @binding(0) var<uniform> face: PointFace;
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
}
struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}
@vertex fn vs(in: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let world = model * vec4<f32>(in.position, 1.0);
    return VertexOutput(face.view_projection * world, world.xyz);
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let distance = length(in.world_position - face.light.xyz) / face.light.w;
    return vec4<f32>(distance, 0.0, 0.0, 1.0);
}
