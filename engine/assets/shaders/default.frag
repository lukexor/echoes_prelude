#version 450

layout (location = 0) in vec4 color;
layout (location = 1) in vec2 uv;

layout (location = 0) out vec4 out_color;

layout (std140, set = 0, binding = 1) uniform SceneBuffer {
  vec4 fog_color;
  vec4 fog_distances;
  vec4 ambient_color;
  vec4 sunlight_direction;
  vec4 sunlight_color;
} scene_data;

void main() {
  out_color = color + scene_data.ambient_color.rgba;
}
