import imgui
from tkinter import filedialog
import util_gau
import glfw
import OpenGL.GL as gl
import imageio
import numpy as np

def gs_elements_control_ui(window, g_renderer, gaussians, g_camera, dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, g_scale_modifier, g_auto_sort, g_renderer_idx, g_renderer_list, g_render_mode, g_render_mode_tables):
    changed = False

    if imgui.begin("Control", True):
        # rendering backend
        changed_backend, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["ogl", "cuda"][:len(g_renderer_list)])
        if changed_backend:
            g_renderer = g_renderer_list[g_renderer_idx]
            update_activated_renderer_state(gaussians)

        imgui.text(f"fps = {imgui.get_io().framerate:.1f}")

        changed_reduce_updates, g_renderer.reduce_updates = imgui.checkbox(
                "reduce updates", g_renderer.reduce_updates,
            )

        imgui.text(f"# of Gaus = {len(gaussians)}")
        if imgui.button(label='open ply'):
            file_path = filedialog.askopenfilename(title="open ply",
                initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                filetypes=[('ply file', '.ply')]
                )
            if file_path:
                try:
                    gaussians = util_gau.load_ply(file_path)  # 移除缩放因子参数
                    gaussians.scale_data(5.0)  # 应用缩放
                    g_renderer.update_gaussian_data(gaussians)
                    g_renderer.set_points_center(gaussians.points_center)
                    g_renderer.sort_and_update(g_camera)
                except RuntimeError as e:
                    pass

        # 添加控制features_dc的滑动条
        changed_dc_scale, new_dc_scale_factor = imgui.slider_float(
            "DC Scale", dc_scale_factor, 0.1, 2.0, "DC Scale Factor = %.2f")
        imgui.same_line()
        if imgui.button(label="Reset DC Scale"):
            new_dc_scale_factor = 1.
            changed_dc_scale = True
        if changed_dc_scale:
            g_renderer.adjust_dc_features(new_dc_scale_factor)
            dc_scale_factor = new_dc_scale_factor

        # 添加控制features_extra的滑动条
        changed_extra_scale, new_extra_scale_factor = imgui.slider_float(
            "Extra Scale", extra_scale_factor, 0.1, 2.0, "Extra Scale Factor = %.2f")
        imgui.same_line()
        if imgui.button(label="Reset Extra Scale"):
            new_extra_scale_factor = 1.
            changed_extra_scale = True
        if changed_extra_scale:
            g_renderer.adjust_extra_features(new_extra_scale_factor)
            extra_scale_factor = new_extra_scale_factor

        # 在Control面板中添加滑动条，对rgb进行变化
        changed_red, g_rgb_factor[0] = imgui.slider_float(
            "Red", g_rgb_factor[0], 0.00, 2.0, "%.4f")
        imgui.same_line()
        if imgui.button(label="Reset Red"):
            g_rgb_factor[0] = 1.
            changed_red = True
        changed_green, g_rgb_factor[1] = imgui.slider_float(
            "Green", g_rgb_factor[1], 0.00, 2.0, "%.4f")
        imgui.same_line()
        if imgui.button(label="Reset Green"):
            g_rgb_factor[1] = 1.
            changed_green = True
        changed_blue, g_rgb_factor[2] = imgui.slider_float(
            "Blue", g_rgb_factor[2], 0.00, 2.0, "%.4f")
        imgui.same_line()
        if imgui.button(label="Reset Blue"):
            g_rgb_factor[2] = 1.
            changed_blue = True
        if changed_red or changed_green or changed_blue:
            # 当任何一个颜色滑动条的值改变时，更新渲染器中的颜色
            g_renderer.update_color_factor(g_rgb_factor)

        # scale modifier
        changed_scale, g_scale_modifier = imgui.slider_float(
            "scale", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
        )
        imgui.same_line()
        if imgui.button(label="Reset scale"):
            g_scale_modifier = 1.
            changed_scale = True

        if changed_scale:
            g_renderer.set_scale_modifier(g_scale_modifier)

        # 在ImGui的控制面板中添加旋转控制的滑动条
        changed_x, g_rot_modifier[0] = imgui.slider_float("Rot X°", g_rot_modifier[0], -180.0, 180.0, "%.1f")
        imgui.same_line()
        if imgui.button("Reset X"):
            g_rot_modifier[0] = 0.0
            changed_x = True
        changed_y, g_rot_modifier[1] = imgui.slider_float("Rot Y°", g_rot_modifier[1], -180.0, 180.0, "%.1f")
        imgui.same_line()
        if imgui.button("Reset Y"):
            g_rot_modifier[1] = 0.0
            changed_y = True
        changed_z, g_rot_modifier[2] = imgui.slider_float("Rot Z°", g_rot_modifier[2], -180.0, 180.0, "%.1f")
        imgui.same_line()
        if imgui.button("Reset Z"):
            g_rot_modifier[2] = 0.0
            changed_z = True
        # 当旋转滑动条的值改变时，或者任何一个轴的重置按钮被点击时
        if changed_x or changed_y or changed_z:
            g_renderer.set_rot_modifier(g_rot_modifier)

        # 添加控制光照旋转的滑动条
        changed_x, g_light_rotation[0] = imgui.slider_float("Light Rot X°", g_light_rotation[0], -180.0, 180.0, "%.1f")
        imgui.same_line()
        if imgui.button("Reset Light X"):
            g_light_rotation[0] = 0.0
            changed_x = True
        changed_y, g_light_rotation[1] = imgui.slider_float("Light Rot Y°", g_light_rotation[1], -180.0, 180.0, "%.1f")
        imgui.same_line()
        if imgui.button("Reset Light Y"):
            g_light_rotation[1] = 0.0
            changed_y = True
        changed_z, g_light_rotation[2] = imgui.slider_float("Light Rot Z°", g_light_rotation[2], -180.0, 180.0, "%.1f")
        imgui.same_line()
        if imgui.button("Reset Light Z"):
            g_light_rotation[2] = 0.0
            changed_z = True
        if changed_x or changed_y or changed_z:
            # 当任何一个轴的旋转角度改变时，更新渲染器中的光照旋转
            g_renderer.set_light_rotation(g_light_rotation)

        # render mode
        changed_render_mode, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
        if changed_render_mode:
            g_renderer.set_render_mod(g_render_mode - 6)

        # sort button
        if imgui.button(label='sort Gaussians'):
            g_renderer.sort_and_update(g_camera)
        imgui.same_line()
        changed_auto_sort, g_auto_sort = imgui.checkbox(
                "auto sort", g_auto_sort,
            )
        if g_auto_sort:
            g_renderer.sort_and_update(g_camera)

        if imgui.button(label='save image'):
            width, height = glfw.get_framebuffer_size(window)
            nrChannels = 3
            stride = nrChannels * width
            stride += (4 - stride % 4) if stride % 4 else 0
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
            gl.glReadBuffer(gl.GL_FRONT)
            bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
            imageio.imwrite("save.png", img[::-1])
            # save intermediate information
            # np.savez(
            #     "save.npz",
            #     gau_xyz=gaussians.xyz,
            #     gau_s=gaussians.scale,
            #     gau_rot=gaussians.rot,
            #     gau_c=gaussians.sh,
            #     gau_a=gaussians.opacity,
            #     viewmat=g_camera.get_view_matrix(),
            #     projmat=g_camera.get_project_matrix(),
            #     hfovxyfocal=g_camera.get_htanfovxy_focal()
            # )
        imgui.end()

    return g_renderer, gaussians, g_camera, dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, g_scale_modifier, g_auto_sort, g_renderer_idx, g_renderer_list, g_render_mode, changed