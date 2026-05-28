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
