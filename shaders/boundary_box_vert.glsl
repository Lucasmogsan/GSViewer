#version 430 core
// layout (location = 0) in vec3 vertexPosition;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 points_center;
uniform mat3 cube_rotation;
uniform vec3 cubeMin; 
uniform vec3 cubeMax; 

void main()
{
    // 立方体的8个角点
    vec3 corners[8] = vec3[8](
        vec3(cubeMin.x, cubeMax.y, cubeMin.z), // near_top_left
        vec3(cubeMax.x, cubeMax.y, cubeMin.z), // near_top_right
        vec3(cubeMax.x, cubeMin.y, cubeMin.z), // near_bottom_right
        vec3(cubeMin.x, cubeMin.y, cubeMin.z), // near_bottom_left
        vec3(cubeMin.x, cubeMax.y, cubeMax.z), // far_top_left
        vec3(cubeMax.x, cubeMax.y, cubeMax.z), // far_top_right
        vec3(cubeMax.x, cubeMin.y, cubeMax.z), // far_bottom_right
        vec3(cubeMin.x, cubeMin.y, cubeMax.z)  // far_bottom_left
    );

    // 根据gl_VertexID选择角点
    vec3 corner = corners[gl_VertexID % 8];

    // 应用旋转和平移
    vec3 worldPosition = points_center + cube_rotation * (corner - points_center);
    vec4 pos_all = projection_matrix * view_matrix * vec4(worldPosition, 1.0);

    // vec4 pos_all2 = projection_matrix * view_matrix * vec4(vertexPosition, 1.0);
    gl_Position = pos_all; 
}

