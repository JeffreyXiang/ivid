#version 130

uniform mat4 u_projection;
uniform mat4 u_modelview;

in vec3 i_position;
in vec3 i_normal;
in vec2 i_texcoord;
in float i_flag;

out vec3 v_position;
out vec3 v_normal;
out vec2 v_texcoord;
out float v_depth;
out float v_is_edge;
out float v_is_padding;
out float v_is_eroded;

void main() {
    gl_Position = u_projection * u_modelview * vec4(i_position, 1.0);
    v_position = i_position;
    v_normal = normalize(i_normal);
    v_texcoord = i_texcoord;
    v_depth = -(u_modelview * vec4(i_position, 1.0)).z;

    v_is_edge = mod(i_flag, 2.0);
    v_is_padding = mod(floor(i_flag / 2.0), 2.0);
    v_is_eroded = mod(floor(i_flag / 4.0), 2.0);
}
