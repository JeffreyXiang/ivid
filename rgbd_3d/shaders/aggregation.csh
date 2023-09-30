#version 430

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, rgba32f) uniform image2D out_color;
layout(binding = 1, rg32f) uniform image2D out_depth;
layout(binding = 2, rg32f) uniform image2D out_mask;

uniform sampler2D colortex;
uniform sampler2D depthtex;

void main()
{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    vec4 color = texelFetch(colortex, pixel, 0);
    float depth = texelFetch(depthtex, pixel, 0).r;

    float weight_color = color.a;
    float weight_depth = color.a > 1e-14 ? 1.0 : color.a > 0.0 ? 1e-8 : 0.0;
    float mask_color = color.a > 1e-6 ? 1.0 : 0.0;
    float mask_depth = color.a > 1e-14 ? 1.0 : 0.0;

    vec4 prev_color = imageLoad(out_color, pixel);
    vec4 prev_depth = imageLoad(out_depth, pixel);
    vec4 prev_mask = imageLoad(out_mask, pixel);

    if (abs(prev_depth.g - 1e-8) < 1e-8 && abs(weight_depth - 1e-8) < 1e-8) {
        if (depth * 1e-8 > prev_depth.r) {
            prev_depth.r = depth * 1e-8;
            prev_depth.g = 1e-8;
            prev_color.rgb = color.rgb * weight_color;
            prev_color.a = weight_color;
        }
    }
    else {
        prev_depth += vec4(depth * weight_depth, weight_depth, 0.0, 0.0);
        prev_color += vec4(color.rgb * weight_color, weight_color);
    }
    prev_mask += vec4(mask_depth, mask_color, 0.0, 0.0);

    imageStore(out_color, pixel, prev_color);
    imageStore(out_depth, pixel, prev_depth);
    imageStore(out_mask, pixel, prev_mask);
}
