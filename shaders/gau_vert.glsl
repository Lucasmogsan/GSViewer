#version 430 core

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

layout(location = 0) in vec2 position;
// layout(location = 1) in vec3 g_pos;
// layout(location = 2) in vec4 g_rot;
// layout(location = 3) in vec3 g_scale;
// layout(location = 4) in vec3 g_dc_color;
// layout(location = 5) in float g_opacity;


#define POS_IDX 0
#define ROT_IDX 3
#define SCALE_IDX 7
#define OPACITY_IDX 10
#define SH_IDX 11

layout (std430, binding=0) buffer gaussian_data {
	float g_data[];
	// compact version of following data
	// vec3 g_pos[];
	// vec4 g_rot[];
	// vec3 g_scale[];
	// float g_opacity[];
	// vec3 g_sh[];
};
layout (std430, binding=1) buffer gaussian_order {
	int gi[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 hfovxy_focal;
uniform vec3 cam_pos;
uniform int sh_dim;
uniform float scale_modifier;
uniform float dc_factor; // 更新DC特征的调整系数并应用所有调整
uniform float extra_factor; // 更新除DC特征外的调整系数并应用所有调整
uniform vec3 color_scale_factors; //调整颜色因子
uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 gaussian
uniform vec4 rot_modifier; // 旋转因子四元数
uniform vec3 light_rotation; // 光照旋转角度，分别对应X、Y、Z轴

// render_boundary
uniform vec3 points_center; 
uniform int enable_aabb; 
uniform int enable_obb; 
uniform mat3 cube_rotation;
uniform vec3 cubeMin; 
uniform vec3 cubeMax; 

out vec3 color;
out float alpha;
out vec3 conic;
out vec2 coordxy;  // local coordinate in quad, unit in pixel
out vec3 approxNormal; // 向片段着色器传递的近似法向量

mat3 computeCov3D(vec3 scale, vec4 q)  // should be correct
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

    mat3 R = mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

    mat3 M = S * R;
    mat3 Sigma = transpose(M) * M;
    return Sigma;
}

vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix)
{
    vec4 t = mean_view;
    // why need this? Try remove this later
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0
    );
    mat3 W = transpose(mat3(viewmatrix));
    mat3 T = W * J;

    mat3 cov = transpose(T) * transpose(cov3D) * T;
    // Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

vec3 get_vec3(int offset)
{
	return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}
vec4 get_vec4(int offset)
{
	return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

// 简单的四元数乘法函数实现
vec4 quatMultiply(vec4 q1, vec4 q2) {
    return vec4(
        q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
        q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
        q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
        q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    );
}


// dir是一个默认初始方向
// rotation是一个三维向量，分别代表绕X、Y、Z轴的旋转角度（以度为单位）
vec3 rotateLightDirection(vec3 dir, vec3 rotation) {
    // 将角度从度转换为弧度
    vec3 radAngles = radians(rotation);

    // 绕X轴旋转
    vec3 rotatedDir;
    rotatedDir.x = dir.x;
    rotatedDir.y = dir.y * cos(radAngles.x) - dir.z * sin(radAngles.x);
    rotatedDir.z = dir.y * sin(radAngles.x) + dir.z * cos(radAngles.x);
    dir = rotatedDir;

    // 绕Y轴旋转
    rotatedDir.x = dir.x * cos(radAngles.y) + dir.z * sin(radAngles.y);
    rotatedDir.y = dir.y;
    rotatedDir.z = -dir.x * sin(radAngles.y) + dir.z * cos(radAngles.y);
    dir = rotatedDir;

    // 绕Z轴旋转
    rotatedDir.x = dir.x * cos(radAngles.z) - dir.y * sin(radAngles.z);
    rotatedDir.y = dir.x * sin(radAngles.z) + dir.y * cos(radAngles.z);
    rotatedDir.z = dir.z;
    dir = rotatedDir;

    return dir;
}

bool isInsideRotatedCube(int enable_aabb, vec3 point, vec3 points_center, vec3 cubeMin, vec3 cubeMax, mat3 rotation) {
    // 如果没有启用任何包围盒功能，直接返回true
    if (enable_aabb == 0 && enable_obb == 0) {
        return true;
    }

    // 如果启用了OBB
    if (enable_obb == 1) {
        vec3 transformed_point = inverse(rotation) * (point - points_center);
        return all(greaterThanEqual(transformed_point, cubeMin)) && all(lessThanEqual(transformed_point, cubeMax));
    }

    // 如果启用了AABB
    if (enable_aabb == 1) {
        vec3 transformed_point = point - points_center;
        vec3 cubeMinPoint = points_center + cubeMin;
        vec3 cubeMaxPoint = points_center + cubeMax;
        return all(greaterThanEqual(transformed_point, cubeMinPoint)) && all(lessThanEqual(transformed_point, cubeMaxPoint));
    }
}

void main()
{
	int boxid = gi[gl_InstanceID];
	int total_dim = 3 + 4 + 3 + 1 + sh_dim;
	int start = boxid * total_dim;
	vec4 g_pos = vec4(get_vec3(start + POS_IDX), 1.f);
	
	// 检查顶点是否在立方体范围内
	bool insideCube = isInsideRotatedCube(enable_aabb, g_pos.xyz, points_center, cubeMin, cubeMax, cube_rotation);
	if (!insideCube)
	{
	    // 如果顶点不在立方体范围内，通过设置特殊的gl_Position来丢弃它
	    gl_Position = vec4(-100, -100, -100, 1);
	    return;
	}

    vec4 g_pos_view = view_matrix * g_pos;
    vec4 g_pos_screen = projection_matrix * g_pos_view;
	g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
    g_pos_screen.w = 1.f;
	// early culling
	if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3))))
	{
		gl_Position = vec4(-100, -100, -100, 1);
		return;
	}
	vec4 g_rot = get_vec4(start + ROT_IDX);
	vec3 g_scale = get_vec3(start + SCALE_IDX);
	float g_opacity = g_data[start + OPACITY_IDX];

    mat3 cov3d = computeCov3D(g_scale * scale_modifier, quatMultiply(g_rot, rot_modifier));
    vec2 wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
    vec3 cov2d = computeCov2D(g_pos_view, 
                              hfovxy_focal.z, 
                              hfovxy_focal.z, 
                              hfovxy_focal.x, 
                              hfovxy_focal.y, 
                              cov3d, 
                              view_matrix);

    // Invert covariance (EWA algorithm)
	float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
	if (det == 0.0f)
		gl_Position = vec4(0.f, 0.f, 0.f, 0.f);
    
    float det_inv = 1.f / det;
	conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    
    vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));  // screen space half quad height and width
    vec2 quadwh_ndc = quadwh_scr / wh * 2;  // in ndc space
    g_pos_screen.xy = g_pos_screen.xy + position * quadwh_ndc;
    coordxy = position * quadwh_scr;
    gl_Position = g_pos_screen;
    
	// 透明度
    alpha = g_opacity;

	//Depth
	if (render_mod == -3)
	{
		float depth = -g_pos_view.z;
		depth = depth < 0.05 ? 1 : depth;
		depth = 1 / depth;
		color = vec3(depth, depth, depth);
		return;
	}

	// 计算指向相机的近似法向量
    approxNormal = normalize(cam_pos - g_pos.xyz);

	//Normal
    if (render_mod == -2) {
        // 计算指向相机的近似法向量
        vec3 viewDirection = normalize(cam_pos - g_pos.xyz);
        // 使用法向量计算颜色，这里简单地将法向量的方向映射到颜色上
        vec3 normalColor = 0.5 * (viewDirection + 1.0); // 将法向量的范围从[-1, 1]映射到[0, 1]
        color = normalColor;
        return;
    }

	// Covert SH to color
	int sh_start = start + SH_IDX;
	vec3 dir = g_pos.xyz - cam_pos;
    dir = normalize(dir);

	// vec3 dir = vec3(1.0, 0.0, 0.0); // 示例方向
	// 使用rotateLightDirection函数来旋转lightDir
	dir = rotateLightDirection(dir, light_rotation);

	// 计算颜色
	// 用球谐系数计算颜色
	// SH_C0 是一个常量，用于调整球谐光照的强度或颜色。
	// 第0阶球谐系数，通常用于表示环境光照的平均颜色
	color = SH_C0 * get_vec3(sh_start);
	
	if (sh_dim > 3 && render_mod >= 1)  // 1 * 3
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		color = color 
				- SH_C1 * y * get_vec3(sh_start + 1 * 3)  // 对应Y方向影响
				+ SH_C1 * z * get_vec3(sh_start + 2 * 3)  // 对应Z方向影响
				- SH_C1 * x * get_vec3(sh_start + 3 * 3); // 对应X方向影响

		// 乘以dc_factor
		color *= dc_factor;

		if (sh_dim > 12 && render_mod >= 2)  // (1 + 3) * 3
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			color = color +
				SH_C2_0 * xy * get_vec3(sh_start + 4 * 3) +
				SH_C2_1 * yz * get_vec3(sh_start + 5 * 3) +
				SH_C2_2 * (2.0f * zz - xx - yy) * get_vec3(sh_start + 6 * 3) +
				SH_C2_3 * xz * get_vec3(sh_start + 7 * 3) +
				SH_C2_4 * (xx - yy) * get_vec3(sh_start + 8 * 3);

			if (sh_dim > 27 && render_mod >= 3)  // (1 + 3 + 5) * 3
			{
				color = color +
					SH_C3_0 * y * (3.0f * xx - yy) * get_vec3(sh_start + 9 * 3) +
					SH_C3_1 * xy * z * get_vec3(sh_start + 10 * 3) +
					SH_C3_2 * y * (4.0f * zz - xx - yy) * get_vec3(sh_start + 11 * 3) +
					SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * get_vec3(sh_start + 12 * 3) +
					SH_C3_4 * x * (4.0f * zz - xx - yy) * get_vec3(sh_start + 13 * 3) +
					SH_C3_5 * z * (xx - yy) * get_vec3(sh_start + 14 * 3) +
					SH_C3_6 * x * (xx - 3.0f * yy) * get_vec3(sh_start + 15 * 3);
			}
			// 乘以额外特征的调整系数
			color *= extra_factor;
		}
	}
	color += 0.5f;

	color *= color_scale_factors; // 将颜色向量的每个分量乘以对应的缩放因子
}
