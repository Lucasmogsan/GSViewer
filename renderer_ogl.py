from OpenGL import GL as gl
import util
import util_gau
import numpy as np
import open3d as o3d


try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except:
    wglSwapIntervalEXT = None


_sort_buffer_xyz = None
_sort_buffer_gausid = None  # used to tell whether gaussian is reloaded

def _sort_gaussian_cpu(gaus, view_mat):
    xyz = np.asarray(gaus.xyz)
    view_mat = np.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index


def _sort_gaussian_cupy(gaus, view_mat):
    import cupy as cp
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = cp.asarray(gaus.xyz)
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = cp.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = cp.argsort(depth)
    index = index.astype(cp.int32).reshape(-1, 1)

    index = cp.asnumpy(index) # convert to numpy
    return index


def _sort_gaussian_torch(gaus, view_mat):
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index


# Decide which sort to use
_sort_gaussian = None
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
    print("Detect torch cuda installed, will use torch as sorting backend")
    _sort_gaussian = _sort_gaussian_torch
except ImportError:
    try:
        import cupy as cp
        print("Detect cupy installed, will use cupy as sorting backend")
        _sort_gaussian = _sort_gaussian_cupy
    except ImportError:
        _sort_gaussian = _sort_gaussian_cpu


class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None
        self._reduce_updates = True

    @property
    def reduce_updates(self):
        return self._reduce_updates

    @reduce_updates.setter
    def reduce_updates(self, val):
        self._reduce_updates = val
        self.update_vsync()

    def update_vsync(self):
        print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        raise NotImplementedError()
    
    def sort_and_update(self):
        raise NotImplementedError()

    def set_scale_modifier(self, modifier: float):
        raise NotImplementedError()
    
    def set_render_mod(self, mod: int):
        raise NotImplementedError()
    
    def update_camera_pose(self, camera: util.Camera):
        raise NotImplementedError()

    def update_camera_intrin(self, camera: util.Camera):
        raise NotImplementedError()
    
    def set_enable_cube(self, enable_cube: int):
        raise NotImplementedError()
    
    def set_cube_rotation(self, cube_rotation: list):
        raise NotImplementedError()

    def set_point_cubeMin(self, point_cubeMin: list):
        raise NotImplementedError()
    
    def set_point_cubeMax(self, point_cubeMax: list):
        raise NotImplementedError()
    
    def draw(self):
        raise NotImplementedError()
    
    def set_render_reso(self, w, h):
        raise NotImplementedError()


class OpenGLRenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        self.program = util.load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')
        self.program_boundary_box = util.load_shaders('shaders/boundary_box_vert.glsl','shaders/boundary_box_frag.glsl')
        
        # Vertex data for a quad
        self.quad_v = np.array([
            -1,  1,
            1,  1,
            1, -1,
            -1, -1
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32).reshape(2, 3)
        
        # load quad geometry
        vao, buffer_id = util.set_attributes(self.program, ["position"], [self.quad_v])
        util.set_faces_tovao(vao, self.quad_f)
        self.vao = vao
        self.ebo = util.set_faces_tovao(self.vao, self.quad_f)
        self.gau_bufferid = None
        self.index_bufferid = None

        # initial box
        self.switch_show_boundary_box = False
        # 创建并绑定顶点数组对象
        self.vao_box = gl.glGenVertexArrays(1)
        self.vbo_box = None
        self.ebo_box = None
        self.box_vertices_count_triangles = 0
        self.box_vertices_count_lines = 0

        # opengl settings
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.update_vsync()

    def __del__(self):
        # 清理资源
        gl.glDeleteBuffers(1, [self.vbo_box])
        gl.glDeleteBuffers(1, [self.ebo_box])
        gl.glDeleteVertexArrays(1, [self.vao_box])

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        else:
            print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        # load gaussian geometry
        gaussian_data = gaus.flat()
        self.gau_bufferid = util.set_storage_buffer_data(self.program, "gaussian_data", gaussian_data, 
                                                         bind_idx=0,
                                                         buffer_id=self.gau_bufferid)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        self.index_bufferid = util.set_storage_buffer_data(self.program, "gi", index, 
                                                           bind_idx=1,
                                                           buffer_id=self.index_bufferid)
        return
   
    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(), "hfovxy_focal")

    # Set the center point coordinates
    def set_points_center(self, points_center: list):
        util.set_uniform_v3(self.program, points_center, "points_center")

    # Set whether to use a cube to limit the rendering area
    def set_enable_cube(self, enable_cube: int):
        util.set_uniform_1int(self.program, enable_cube, "enable_cube")

    # Set the rotation of the cube
    def set_cube_rotation(self, cube_rotation: list):
        R = util.calculate_rotation_matrix(cube_rotation)
        util.set_uniform_mat3(self.program, R, "cube_rotation") 

    # Set the minimum coordinates of the cube
    def set_point_cubeMin(self, point_cubeMin: list):
        util.set_uniform_v3(self.program, point_cubeMin, "cubeMin")

    # Set the maximum coordinates of the cube
    def set_point_cubeMax(self, point_cubeMax: list):
        util.set_uniform_v3(self.program, point_cubeMax, "cubeMax")

    def draw_boundary_box(self, points_center: list, point_cubeMin, point_cubeMax, cube_rotation, camera: util.Camera):
        R = util.calculate_rotation_matrix(cube_rotation)
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_project_matrix()

        # 设置uniforms
        util.set_uniform_mat4(self.program_boundary_box, view_mat, "view_matrix")
        util.set_uniform_mat4(self.program_boundary_box, proj_mat, "projection_matrix")
        util.set_uniform_mat3(self.program_boundary_box, R, "cube_rotation") 
        util.set_uniform_v3(self.program_boundary_box, points_center, "points_center")
        util.set_uniform_v3(self.program_boundary_box, point_cubeMin, "cubeMin")
        util.set_uniform_v3(self.program_boundary_box, point_cubeMax, "cubeMax")

        vertices, indices_triangles, indices_lines = util.create_box_mesh_from_bounds(points_center, point_cubeMin, point_cubeMax)
        rotated_vertices = np.dot(vertices, R.T) 
        vertices = rotated_vertices.flatten()

        # 记录索引数量，用于绘制
        self.box_vertices_count_triangles = len(indices_triangles)
        self.box_vertices_count_lines = len(indices_lines)

        # # 使用 set_attribute 函数设置顶点属性
        # vao, buffer_id = util.set_attribute(
        #             self.program_boundary_box, 
        #             'vertexPosition', 
        #             vertices, 
        #             3,
        #             self.vao_box
        #         )
        # self.vbo_box = buffer_id

        # 使用 set_faces_tovao 函数设置索引缓冲对象，并接收返回的 ebo
        # 设置三角形索引缓冲区
        self.ebo_box_triangles = util.set_faces_tovao(self.vao_box, indices_triangles.flatten())
        # 设置线框索引缓冲区
        self.ebo_box_lines = util.set_faces_tovao(self.vao_box, indices_lines.flatten())


    def toggle_draw_boundary_box(self):
        # 切换包围盒显示状态
        self.switch_show_boundary_box = not self.switch_show_boundary_box

    def draw(self):
        # 主渲染高斯
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER,self.ebo)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)

        # 绘制包围盒
        if self.switch_show_boundary_box:
            if self.vao_box and self.ebo_box_triangles and self.ebo_box_lines:
                # 绑定着色器程序
                gl.glUseProgram(self.program_boundary_box)
                gl.glBindVertexArray(self.vao_box)
                # 使用填充模式绘制面
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_box_triangles)
                gl.glDrawElements(gl.GL_TRIANGLES, self.box_vertices_count_triangles, gl.GL_UNSIGNED_INT, None)
                # 使用线框模式绘制线框
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_box_lines)
                gl.glDrawElements(gl.GL_LINES, self.box_vertices_count_lines, gl.GL_UNSIGNED_INT, None)
                # 解绑 VAO 和 VBO
                gl.glBindVertexArray(0)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
