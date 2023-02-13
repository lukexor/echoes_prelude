#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec4 color;
layout (location = 3) in vec2 in_uv;

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec2 out_uv;

layout (set = 0, binding = 0) uniform CameraBuffer {
  mat4 projection;
  mat4 view;
  mat4 projection_view;
} cb;

struct ObjectData {
	mat4 transform;
};
layout (std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
	ObjectData objects[];
} ob;

void main() {
	mat4 transform = ob.objects[gl_BaseInstance].transform;
	gl_Position = cb.projection_view * transform * vec4(position, 1.0);
	out_color = color;
	out_uv = in_uv;
}
