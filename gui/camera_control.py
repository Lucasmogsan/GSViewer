import imgui
import numpy as np

def camera_control_ui(g_camera, g_show_camera_control):
    if g_show_camera_control:
        if imgui.begin("Camera Control", True):  # 添加窗口标题和可关闭参数
            imgui.push_item_width(150)  # 设置滑动条宽度

            if imgui.button(label='Rot 180'):
                g_camera.flip_ground()

            changed, g_camera.use_free_rotation = imgui.checkbox(
                "Use Free Rotation", g_camera.use_free_rotation
                )

            changed, g_camera.use_custom_rotation_center = imgui.checkbox(
                "Use Custom Rotation Center", g_camera.use_custom_rotation_center
                )
            if g_camera.use_custom_rotation_center:
                changed, g_camera.rotation_center[0] = imgui.input_float("Rotation Center X", g_camera.rotation_center[0])
                changed, g_camera.rotation_center[1] = imgui.input_float("Rotation Center Y", g_camera.rotation_center[1])
                changed, g_camera.rotation_center[2] = imgui.input_float("Rotation Center Z", g_camera.rotation_center[2])

            changed, g_camera.fovy = imgui.slider_float(
                "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="Reset fov"):
                g_camera.fovy = np.pi / 2
                changed = True
            if changed:
                g_camera.is_intrin_dirty = True

            changed, g_camera.target_dist = imgui.slider_float(
                    "t", g_camera.target_dist, 1., 10., "target dist = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset t"):
                g_camera.target_dist = 5.0
                changed = True  # 确保重置按钮也能触发更新
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.sensitivities['rot'] = imgui.slider_float(
                    "r", g_camera.sensitivities['rot'], 0.002, 0.1, "rotate speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset r"):
                g_camera.sensitivities['rot'] = 0.02

            changed, g_camera.sensitivities['trans'] = imgui.slider_float(
                    "m", g_camera.sensitivities['trans'], 0.001, 0.03, "move speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset m"):
                g_camera.sensitivities['trans'] = 0.01

            changed, g_camera.sensitivities['zoom'] = imgui.slider_float(
                    "z", g_camera.sensitivities['zoom'], 0.001, 1.0, "zoom speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset z"):
                g_camera.sensitivities['zoom'] = 0.08

            changed, g_camera.sensitivities['roll'] = imgui.slider_float(
                    "ro", g_camera.sensitivities['roll'], 0.003, 0.1, "roll speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset ro"):
                g_camera.sensitivities['roll'] = 0.03    

            
            imgui.pop_item_width()  # 恢复滑动条默认宽度

        imgui.end()  # 添加结束窗口