#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 fragColor;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform mat4 model_matrix;

void main()
{
    fragColor = color;
    gl_Position = projection_matrix * view_matrix * model_matrix * vec4(position, 1.0);
}