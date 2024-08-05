#version 430 core

in vec3 color;
in float alpha;
in vec3 conic;
in vec2 coordxy;  // local coordinate in quad, unit in pixel

uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 flat ball, -4 gaussian ball

in vec3 approxNormal; // 从顶点着色器接收的近似法向量

out vec4 FragColor;

void main()
{
    if (render_mod == -4)
    {
        FragColor = vec4(color, 1.f);
        return;
    }

    // Billboard Normal
    if (render_mod == -1)
    {
        // 直接将法向量映射到颜色上
        vec3 normalColor = 0.5 * (normalize(approxNormal) + 1.0);
        FragColor = vec4(normalColor, 1.0);
        
        // 直接将法向量映射到颜色上，并调整亮度以实现平滑过渡
        // vec3 normal = normalize(approxNormal);
        // float intensity = 0.5 * (normal.z + 1.0); // 使用Z分量调整亮度
        // vec3 normalColor = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), intensity); // 使用蓝色到红色的渐变
        // FragColor = vec4(normalColor, 1.0);
        return;
    }

    float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
    if (power > 0.f)
        discard;
    float opacity = min(0.99f, alpha * exp(power));
    if (opacity < 1.f / 255.f)
        discard;
    FragColor = vec4(color, opacity);

    // handling special shading effect
    if (render_mod == -5) //Flat Ball
        FragColor.a = FragColor.a > 0.22 ? 1 : 0;
    else if (render_mod == -6) //Gaussian Ball
    {
        FragColor.a = FragColor.a > 0.22 ? 1 : 0;
        FragColor.rgb = FragColor.rgb * exp(power);
    }
}
