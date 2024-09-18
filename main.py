import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
from gui.camera_control import camera_control_ui
from gui.gs_elements_control import gs_elements_control_ui
from gui.help_content import help_window_ui

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

g_camera = util.Camera(720, 1280)
BACKEND_OGL=0
BACKEND_CUDA=1
g_renderer_list = [
    None, # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = False
g_show_gs_elements_control = True
g_show_help_control = False
g_show_camera_control = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "Normal", "Billboard Normal", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 9


# Initialize rendering boundary related variables
g_show_render_boundary_control = True
use_axis_for_rotation = False 
g_enable_render_boundary_aabb = 0
g_cube_rotation = [0.0, 0.0, 0.0]
g_cube_min = [-4.0, -4.0, -4.0]
g_cube_max = [4.0, 4.0, 4.0]
tmp_cube_rotation = [0.0, 0.0, 0.0]
tmp_cube_min = [-4.0, -4.0, -4.0]
tmp_cube_max = [4.0, 4.0, 4.0]

dc_scale_factor = 1.0  # 直流特征的缩放因子
extra_scale_factor = 1.0  # 额外特征的缩放因子
g_rgb_factor = [1.0, 1.0, 1.0]# 在rgb全局变量区域添加
g_rot_modifier = [0.0, 0.0, 0.0] #设置一个旋转修正因子，默认是1。
g_light_rotation = [0.0, 0.0, 0.0]  # 光照旋转角度，初始化为0

def impl_glfw_init():
    window_name = "GSViewer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_rot_modifier(g_rot_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)

def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_gs_elements_control, g_show_help_control, g_show_camera_control, \
        g_render_mode, g_render_mode_tables, \
        dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, \
        g_show_render_boundary_control, g_enable_render_boundary_aabb, use_axis_for_rotation, g_cube_min, g_cube_max, g_cube_rotation, tmp_cube_min, tmp_cube_max, tmp_cube_rotation
        
    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    try:
        from renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        g_renderer_idx = BACKEND_OGL
    else:
        g_renderer_idx = BACKEND_CUDA

    g_renderer = g_renderer_list[g_renderer_idx]

    # gaussian data
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)
    
    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        
        g_renderer.draw()

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_gs_elements_control = imgui.menu_item(
                    "Show GS Elements Control", None, g_show_gs_elements_control
                )
                clicked, g_show_render_boundary_control = imgui.menu_item(
                    "Show Render Boundary", None, g_show_render_boundary_control
                )
                clicked, g_show_camera_control = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_control
                )
                clicked, g_show_help_control = imgui.menu_item(
                    "Show Help", None, g_show_help_control
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_gs_elements_control:
            g_renderer, gaussians, g_camera, dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, g_scale_modifier, g_auto_sort, g_renderer_idx, g_renderer_list, g_render_mode, changed = gs_elements_control_ui(
                window, g_renderer, gaussians, g_camera, dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, g_scale_modifier, g_auto_sort, g_renderer_idx, g_renderer_list, g_render_mode, g_render_mode_tables
            )

        # Add the following code in the main function or appropriate GUI rendering section
        if g_show_render_boundary_control:
            if imgui.begin("3DGS Render Boundary", True):
                # Add a checkbox to control the enabling of render boundaries for AABB
                changed_aabb, new_enable_aabb = imgui.checkbox("Enable Render Boundary AABB", g_enable_render_boundary_aabb == 1)
                if changed_aabb:
                    if new_enable_aabb:
                        g_enable_render_boundary_aabb = 1
                        g_enable_render_boundary_obb = 0
                        g_renderer.set_enable_aabb(g_enable_render_boundary_aabb)  # 启用AABB
                        g_renderer.set_enable_obb(g_enable_render_boundary_obb)  # 禁用OBB
                    else:
                        g_enable_render_boundary_aabb = 0
                        g_enable_render_boundary_obb = 0
                        g_renderer.set_enable_aabb(g_enable_render_boundary_aabb)  # 禁用AABB
                        g_renderer.set_enable_obb(g_enable_render_boundary_obb)  # 禁用OBB
                        use_axis_for_rotation = False  # 禁用时重置旋转轴使用状态
                        g_cube_rotation = [0.0, 0.0, 0.0]  # 重置为默认角度

                    # Update the renderer for AABB
                    g_cube_min, g_cube_max, _ = gaussians.compute_aabb
                    tmp_cube_min = g_cube_min.copy()  # 创建独立副本
                    tmp_cube_max = g_cube_max.copy()  # 创建独立副本

                    g_renderer.clear_boundary_box()  # 清除之前的边界框资源
                    g_renderer.toggle_draw_boundary_box()
                    g_renderer.set_cube_rotation(g_cube_rotation)
                    g_renderer.set_point_cubeMin(g_cube_min)
                    g_renderer.set_point_cubeMax(g_cube_max)
                    # update_activated_renderer_state(gaussians)

                # Only show "Use Axis for Rotation" checkbox if AABB is enabled
                if g_enable_render_boundary_aabb:
                    imgui.same_line()
                    changed_use_axis, use_axis_for_rotation = imgui.checkbox("Toggle OBB Rotation", use_axis_for_rotation)
                    if changed_use_axis:
                        if use_axis_for_rotation:
                            g_enable_render_boundary_obb = 1
                            g_renderer.set_enable_obb(g_enable_render_boundary_obb) 
                            g_cube_rotation = util.convert_rotation_matrix_to_euler_angles(gaussians.compute_obb[2])
                            tmp_cube_rotation = g_cube_rotation.copy()
                            # 更新边界框的角点为OBB的角点
                            # g_cube_min = g_cube_min*g_cube_rotation
                            # g_cube_max = g_cube_max*g_cube_rotation
                        else:
                            # 重置aabb初始情况
                            g_enable_render_boundary_obb = 0
                            g_renderer.set_enable_obb(g_enable_render_boundary_obb) 
                            g_cube_rotation = [0.0, 0.0, 0.0]
                            tmp_cube_rotation = [0.0, 0.0, 0.0]
                            # g_cube_min, g_cube_max, _ = gaussians.compute_aabb
                        g_renderer.set_cube_rotation(g_cube_rotation)

                # When render boundary is enabled, display sliders to control the cube boundaries
                if g_enable_render_boundary_aabb:
                    fixed_min_xmin = tmp_cube_min[0]
                    fixed_min_xmax = tmp_cube_max[0]
                    fixed_min_ymin = tmp_cube_min[1]
                    fixed_min_ymax = tmp_cube_max[1]
                    fixed_min_zmin = tmp_cube_min[2]
                    fixed_min_zmax = tmp_cube_max[2]

                    fixed_max_xmin = tmp_cube_min[0]
                    fixed_max_xmax = tmp_cube_max[0]
                    fixed_max_ymin = tmp_cube_min[1]
                    fixed_max_ymax = tmp_cube_max[1]
                    fixed_max_zmin = tmp_cube_min[2]
                    fixed_max_zmax = tmp_cube_max[2] 

                    changed_min_x, g_cube_min[0] = imgui.slider_float("Min X", g_cube_min[0], fixed_min_xmin, fixed_min_xmax, "%.2f")
                    imgui.same_line()
                    if imgui.button("Reset Min X"):
                        g_cube_min[0] = tmp_cube_min[0]
                        changed_min_x = True
                    changed_min_y, g_cube_min[1] = imgui.slider_float("Min Y", g_cube_min[1], fixed_min_ymin, fixed_min_ymax, "%.2f")
                    imgui.same_line()
                    if imgui.button("Reset Min Y"):
                        g_cube_min[1] = tmp_cube_min[1]
                        changed_min_y = True
                    changed_min_z, g_cube_min[2] = imgui.slider_float("Min Z", g_cube_min[2], fixed_min_zmin, fixed_min_zmax, "%.2f")
                    imgui.same_line()
                    if imgui.button("Reset Min Z"):
                        g_cube_min[2] = tmp_cube_min[2]
                        changed_min_z = True

                    changed_max_x, g_cube_max[0] = imgui.slider_float("Max X", g_cube_max[0], fixed_max_xmin, fixed_max_xmax, "%.2f")
                    imgui.same_line()
                    if imgui.button("Reset Max X"):
                        g_cube_max[0] = tmp_cube_max[0]
                        changed_max_x = True
                    changed_max_y, g_cube_max[1] = imgui.slider_float("Max Y", g_cube_max[1], fixed_max_ymin, fixed_max_ymax, "%.2f")
                    imgui.same_line()
                    if imgui.button("Reset Max Y"):
                        g_cube_max[1] = tmp_cube_max[1]
                        changed_max_y = True
                    changed_max_z, g_cube_max[2] = imgui.slider_float("Max Z", g_cube_max[2], fixed_max_zmin, fixed_max_zmax, "%.2f")
                    imgui.same_line()
                    if imgui.button("Reset Max Z"):
                        g_cube_max[2] = tmp_cube_max[2]
                        changed_max_z = True

                    # Add rotation controls
                    changed_rot_x, g_cube_rotation[0] = imgui.slider_float("Rotate X", g_cube_rotation[0], -180.0, 180.0, "%.2f degrees")
                    imgui.same_line()
                    if imgui.button("Reset Rotate X"):
                        g_cube_rotation[0] = tmp_cube_rotation[0]
                        changed_rot_x = True
                    changed_rot_y, g_cube_rotation[1] = imgui.slider_float("Rotate Y", g_cube_rotation[1], -180.0, 180.0, "%.2f degrees")
                    imgui.same_line()
                    if imgui.button("Reset Rotate Y"):
                        g_cube_rotation[1] = tmp_cube_rotation[1]
                        changed_rot_y = True
                    changed_rot_z, g_cube_rotation[2] = imgui.slider_float("Rotate Z", g_cube_rotation[2], -180.0, 180.0, "%.2f degrees")
                    imgui.same_line()
                    if imgui.button("Reset Rotate Z"):
                        g_cube_rotation[2] = tmp_cube_rotation[2]
                        changed_rot_z = True
                    
                    # If any changes occurred, update the OpenGL renderer's cube boundaries
                    if changed_rot_x or changed_rot_y or changed_rot_z or changed_min_x or changed_min_y or changed_min_z or changed_max_x or changed_max_y or changed_max_z:
                        g_renderer.set_cube_rotation(g_cube_rotation)
                        g_renderer.set_point_cubeMin(g_cube_min)
                        g_renderer.set_point_cubeMax(g_cube_max)
                    
                    # 绘制边界框
                    if g_enable_render_boundary_aabb:
                        g_renderer.draw_boundary_box(gaussians.points_center, g_cube_min, g_cube_max, g_cube_rotation, g_camera)

                # 如果AABB启用，则显示导出按钮和路径选择
                if g_enable_render_boundary_aabb:
                    # 导出文件路径行
                    imgui.text("Export file path:")
                    changed, export_path = imgui.input_text("##exportpath", export_path, 256)
                    imgui.same_line()  # 确保浏览按钮在文本框同一行显示
                    if imgui.button("Browse##export"):
                        root = tk.Tk()
                        root.withdraw()  # 隐藏Tkinter主窗口
                        selected_path = filedialog.asksaveasfilename(
                            parent=root,
                            title="Select export file",
                            filetypes=[('PLY Files', '*.ply')],
                            defaultextension='.ply'
                        )
                        if selected_path:  # 确保用户选择了文件
                            export_path = selected_path
                            export_status = "File selected"  # 更新状态为文件已选择

                    # 导出按钮
                    if imgui.button("Export"):
                        if export_path:  # 确保路径存在才可以使用
                            # 调用export_ply函数，传入export_path和其他必要参数
                            success = util_gau.export_ply(gaussians, export_path, g_enable_render_boundary_aabb, g_enable_render_boundary_obb, g_cube_min, g_cube_max, g_cube_rotation)
                            if success:
                                export_status = "Export successful"  # 设置状态为成功
                            else:
                                export_status = "Export failed"  # 设置状态为失败
                        else:
                            export_status = "No file path selected"  # 设置状态为未选择文件路径
                    # 更新状态显示
                    imgui.same_line()  # 确保状态信息在文本框同一行显示
                    imgui.text(export_status)  # 显示当前的导出状态
                else:
                    export_path = ""  # 确保当AABB未启用时清空路径
                    export_status = "Please select export file path"  # 重置状态为默认文本

                imgui.end()

        # 相机控制（包括相机旋转、平移、缩放）
        camera_control_ui(g_camera, g_show_camera_control)

        # 帮助内容
        help_window_ui(g_show_help_control)
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="3DGS viewer with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()

    main()
