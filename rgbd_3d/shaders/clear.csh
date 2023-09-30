#version 430

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, rgba32f) uniform image2D out_color;
layout(binding = 1, rg32f) uniform image2D out_depth;
layout(binding = 2, rg32f) uniform image2D out_mask;


void main()
{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    imageStore(out_color, pixel, vec4(0.0));
    imageStore(out_depth, pixel, vec4(0.0));
    imageStore(out_mask, pixel, vec4(0.0));
}
