#version 130

uniform sampler2D colortex;

in vec2 v_texcoord;
in float v_is_edge;

out vec4 f_color;

void main() {
    vec3 color = texture(colortex, v_texcoord).rgb;

    if (!gl_FrontFacing) {
        f_color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    f_color = vec4(color, v_is_edge > 0.999 ? 0.0 : 1.0);
}
