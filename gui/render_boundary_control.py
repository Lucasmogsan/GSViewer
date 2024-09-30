import imgui
import tkinter as tk
from tkinter import filedialog
import util
import util_gau

def render_boundary_control_ui(g_renderer, gaussians, g_enable_render_boundary_aabb, g_enable_render_boundary_obb, g_cube_min, g_cube_max, g_cube_rotation, tmp_cube_min, tmp_cube_max, tmp_cube_rotation, use_axis_for_rotation, export_path, export_status):
    if imgui.begin("3DGS Render Boundary", True):
        imgui.push_item_width(150)  # 设置滑动条宽度
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
            # imgui.same_line()
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
                g_renderer.draw_boundary_box(gaussians.points_center, g_cube_min, g_cube_max, g_cube_rotation)

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

        imgui.pop_item_width()  # 恢复滑动条默认宽度
        imgui.end()

    return g_enable_render_boundary_aabb, g_enable_render_boundary_obb, g_cube_min, g_cube_max, g_cube_rotation, tmp_cube_min, tmp_cube_max, tmp_cube_rotation, use_axis_for_rotation, export_path, export_status