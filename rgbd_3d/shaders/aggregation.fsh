#version 130

uniform vec3 u_sample_camera;
// uniform vec3 u_camera;

uniform sampler2D colortex;

in vec3 v_position;
in vec3 v_normal;
in vec2 v_texcoord;
in float v_depth;
in float v_is_edge;
in float v_is_padding;
in float v_is_eroded;

out vec4 f_color;
out vec4 f_depth;

void main() {
    vec3 color = texture(colortex, v_texcoord).rgb;

    if (!gl_FrontFacing) {
        if (v_is_padding > 0.001) discard;
        f_color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    vec3 dir = normalize(u_sample_camera - v_position);
    vec3 normal = normalize(v_normal);
    float weight;
    weight = clamp(dot(dir, normal), 0.0, 1.0);
    weight = acos(weight);
    weight = max(-weight * 20, -50);
    weight = exp(weight);
    weight = max(weight, 1e-4);
    if (v_is_eroded < 0.999) {
        // if (length(u_sample_camera.xy) < 0.01) {
        //     weight *= 1e4;
        // }
    }
    else {
        weight *= 1e-8;
    }

    if (v_is_padding > 0.001 || v_is_edge > 0.999) {
        weight = 1e-16;
    }

    weight = max(weight, 1e-16);

    f_color = vec4(color, weight);
}
