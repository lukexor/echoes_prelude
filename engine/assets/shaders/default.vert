#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;
layout (location = 3) in vec2 uv;

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec2 out_uv;

layout (set = 0, binding = 0) uniform CameraBuffer {
	mat4 projection;
	mat4 view;
} cb;

struct ObjectData {
	mat4 model;
};

layout (std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
	ObjectData objects[];
} ob;

void main() {
	mat4 model = ob.objects[gl_BaseInstance].model;
	gl_Position = cb.projection * cb.view * model * vec4(position, 1.0);
	out_color = color;
	out_uv = uv;
}
