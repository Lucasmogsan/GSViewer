import imgui
import numpy as np

def camera_control_ui(g_camera, g_show_camera_control):
    if g_show_camera_control:
        if imgui.begin("Camera Control", True):  # 添加窗口标题和可关闭参数
            if imgui.button(label='rot 180'):
                g_camera.flip_ground()

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
                    "t", g_camera.target_dist, 1., 8., "target dist = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset t"):
                g_camera.target_dist = 3.0
                changed = True  # 确保重置按钮也能触发更新
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                    "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset r"):
                g_camera.rot_sensitivity = 0.02

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                    "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                    "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                    "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="Reset ro"):
                g_camera.roll_sensitivity = 0.03    
                
        imgui.end()  # 添加结束窗口