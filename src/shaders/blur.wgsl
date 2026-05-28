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
