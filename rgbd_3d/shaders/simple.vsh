#version 130

uniform mat4 u_projection;
uniform mat4 u_modelview;

in vec3 i_position;
in vec2 i_texcoord;
in float i_flag;

out vec2 v_texcoord;
out float v_is_edge;

void main() {
    gl_Position = u_projection * u_modelview * vec4(i_position, 1.0);
    v_texcoord = i_texcoord;
    v_is_edge = mod(i_flag, 2.0);
}
