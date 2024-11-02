import numpy as np
from plyfile import PlyElement, PlyData
from dataclasses import dataclass, field
import scipy as sp
import util
import argparse
from tools.gsconverter.main import gsconverter
import pandas as pd

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    path: str = None
    _original_data: dict = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        # 在初始化后存储原始数据的深拷贝
        self._original_data['xyz'] = np.copy(self.xyz)
        self._original_data['rot'] = np.copy(self.rot)
        self._original_data['scale'] = np.copy(self.scale)
        self._original_data['opacity'] = np.copy(self.opacity)
        self._original_data['sh'] = np.copy(self.sh)

    def __len__(self):
        return len(self.xyz)
    
    def __getitem__(self, idx):
        return GaussianData(
            xyz=self.xyz[idx],
            rot=self.rot[idx],
            scale=self.scale[idx],
            opacity=self.opacity[idx],
            sh=self.sh[idx]
        )
    
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def scale_data(self, scale_to_interval):
        df = pd.DataFrame(self.xyz, columns=['x', 'y', 'z'])
        min_xyz = df.min()
        max_xyz = df.max()
        center_xyz = (min_xyz + max_xyz) / 2
        max_extent = (max_xyz - min_xyz).max()
        scale_factor = scale_to_interval / max_extent
        self.xyz = ((df - center_xyz) * scale_factor).values
        self.rot = self.rot / np.linalg.norm(self.rot, axis=-1, keepdims=True)  # 如果rot是法线
        self.scale *= scale_factor

    @property 
    def restore_original_state(self):
        # 恢复所有数据到其原始状态
        self.xyz = np.copy(self._original_data['xyz'])
        self.rot = np.copy(self._original_data['rot'])
        self.scale = np.copy(self._original_data['scale'])
        self.opacity = np.copy(self._original_data['opacity'])
        self.sh = np.copy(self._original_data['sh'])

    @property 
    def get_original_state(self):
        # 返回一个新的GaussianData实例，其数据为原始数据的副本
        return GaussianData(
            xyz=np.copy(self._original_data['xyz']),
            rot=np.copy(self._original_data['rot']),
            scale=np.copy(self._original_data['scale']),
            opacity=np.copy(self._original_data['opacity']),
            sh=np.copy(self._original_data['sh'])
        )

    @property 
    def sh_dim(self):
        return self.sh.shape[-1]
    
    @property
    def points_center(self):
        return np.mean(self.xyz, axis=0)

    @property
    def points_min(self):
        return np.min(self.xyz, axis=0)

    @property
    def points_max(self):
        return np.max(self.xyz, axis=0)

    @property
    def points_extent(self):
        return self.points_max - self.points_min

    @property
    def compute_aabb(self):
        """计算并返回AABB的八个角点"""
        xmin, ymin, zmin = self.points_min
        xmax, ymax, zmax = self.points_max
        # 计算AABB的八个角点
        aabb_corners =  np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax]
        ])
        return self.points_min, self.points_max, aabb_corners

    @property
    def compute_obb(self):
        """计算并返回OBB的八个角点"""
        df = pd.DataFrame(self.xyz, columns=['x', 'y', 'z'])
        center = df.mean()
        centered_points = df - center
        covariance_matrix = centered_points.cov()
        U, S, Vt = np.linalg.svd(covariance_matrix)
        projected_points = centered_points @ U
        min_bounds = projected_points.min()
        max_bounds = projected_points.max()
        obb_min = center + min_bounds @ U.T
        obb_max = center + max_bounds @ U.T

        # 计算OBB的八个角点
        obb_corners = np.array([
            [obb_min[0], obb_min[1], obb_min[2]],
            [obb_max[0], obb_min[1], obb_min[2]],
            [obb_min[0], obb_max[1], obb_min[2]],
            [obb_max[0], obb_max[1], obb_min[2]],
            [obb_min[0], obb_min[1], obb_max[2]],
            [obb_max[0], obb_min[1], obb_max[2]],
            [obb_min[0], obb_max[1], obb_max[2]],
            [obb_max[0], obb_max[1], obb_max[2]]
        ])
        return obb_min.values, obb_max.values, U, obb_corners

    def _apply_transformations(self):
        transformed_xyz = np.zeros_like(self.xyz)
        for i in range(len(self.xyz)):
            # 将四元数转换为旋转矩阵
            rotation_matrix = sp.spatial.transform.Rotation.from_quat(self.rot[i]).as_matrix()
            # 应用旋转和缩放
            transformed_xyz[i] = rotation_matrix @ (self.xyz[i] * self.scale[i])
        return transformed_xyz

def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    return GaussianData(
        gau_xyz,
        gau_rot,
        gau_s,
        gau_a,
        gau_c
    )

# # 未使用pandas加载速度较慢
# def load_ply(path):
#     max_sh_degree = 3
#     plydata = PlyData.read(path)
#     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                     np.asarray(plydata.elements[0]["y"]),
#                     np.asarray(plydata.elements[0]["z"])),  axis=1)
#     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

#     features_dc = np.zeros((xyz.shape[0], 3, 1))
#     features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
#     features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
#     features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

#     extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
#     extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
#     assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
#     features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
#     for idx, attr_name in enumerate(extra_f_names):
#         features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
#     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
#     features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
#     features_extra = np.transpose(features_extra, [0, 2, 1])

#     scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
#     scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
#     scales = np.zeros((xyz.shape[0], len(scale_names)))
#     for idx, attr_name in enumerate(scale_names):
#         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

#     rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
#     rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
#     rots = np.zeros((xyz.shape[0], len(rot_names)))
#     for idx, attr_name in enumerate(rot_names):
#         rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

#     # pass activate function
#     xyz = xyz.astype(np.float32)
#     rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
#     rots = rots.astype(np.float32)
#     scales = np.exp(scales)
#     scales = scales.astype(np.float32)
#     opacities = 1/(1 + np.exp(- opacities))  # sigmoid
#     opacities = opacities.astype(np.float32)
#     shs = np.concatenate([features_dc.reshape(-1, 3), 
#                         features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
#     shs = shs.astype(np.float32)
#     return GaussianData(xyz, rots, scales, opacities, shs, path=path)

# 使用pandas加载ply速度较快
def load_ply(path):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    
    # 创建一个包含所有相关数据的 DataFrame
    df = pd.DataFrame({
        'x': np.asarray(plydata.elements[0]["x"]),
        'y': np.asarray(plydata.elements[0]["y"]),
        'z': np.asarray(plydata.elements[0]["z"]),
        'opacity': np.asarray(plydata.elements[0]["opacity"]),
        'f_dc_0': np.asarray(plydata.elements[0]["f_dc_0"]),
        'f_dc_1': np.asarray(plydata.elements[0]["f_dc_1"]),
        'f_dc_2': np.asarray(plydata.elements[0]["f_dc_2"])
    })
    
    xyz = df[['x', 'y', 'z']].values
    opacities = df['opacity'].values[..., np.newaxis]
    features_dc = df[['f_dc_0', 'f_dc_1', 'f_dc_2']].values.reshape(-1, 3, 1)

    properties = plydata.elements[0].properties
    property_names = {p.name: p for p in properties}

    extra_f_names = sorted(
        (name for name in property_names if name.startswith("f_rest_")),
        key=lambda x: int(x.split('_')[-1])
    )
    scale_names = sorted(
        (name for name in property_names if name.startswith("scale_")),
        key=lambda x: int(x.split('_')[-1])
    )
    rot_names = sorted(
        (name for name in property_names if name.startswith("rot")),
        key=lambda x: int(x.split('_')[-1])
    )

    all_features = pd.DataFrame({
        **{name: np.asarray(plydata.elements[0][name]) for name in extra_f_names},
        **{name: np.asarray(plydata.elements[0][name]) for name in scale_names},
        **{name: np.asarray(plydata.elements[0][name]) for name in rot_names}
    })

    # 处理 features_extra
    features_extra = all_features[extra_f_names].values
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    # 处理 scales 和 rots
    scales = all_features[scale_names].values
    rots = all_features[rot_names].values

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales).astype(np.float32)
    opacities = (1 / (1 + np.exp(-opacities))).astype(np.float32)  # sigmoid
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)

    return GaussianData(xyz, rots, scales, opacities, shs, path=path)


# def is_inside_rotated_cube(enable_aabb, enable_obb, point, points_center, cube_min, cube_max, rotation_matrix):
#     if enable_aabb == 0 and enable_obb == 0:
#         return True
#     if enable_obb == 1:
#         transformed_point = np.dot(np.linalg.inv(rotation_matrix), (point - points_center))
#         return np.all(transformed_point >= cube_min) and np.all(transformed_point <= cube_max)
#     if enable_aabb == 1:
#         transformed_point = point - points_center
#         cube_min_point = points_center + cube_min
#         cube_max_point = points_center + cube_max
#         return np.all(transformed_point >= cube_min_point) and np.all(transformed_point <= cube_max_point)
#     return False
# 
# def export_ply(gaussian_data, path, enable_aabb, enable_obb, cube_min, cube_max, cube_rotation):
#     rotation_matrix = util.convert_euler_angles_to_rotation_matrix(cube_rotation)
#     points_center = np.mean(gaussian_data.xyz, axis=0)  # Assuming center is the mean of points

#     # Filter points inside the cube
#     mask = np.array([is_inside_rotated_cube(enable_aabb, enable_obb, point, points_center, cube_min, cube_max, rotation_matrix) for point in gaussian_data.xyz])
    
#     # 获取原始数据的副本
#     original_data = gaussian_data.get_original_state

#     # Apply the mask to all attributes of the GaussianData
#     filtered_xyz = original_data.xyz[mask]
#     filtered_rot = original_data.rot[mask]
#     filtered_scale = original_data.scale[mask]
#     filtered_opacity = original_data.opacity[mask]
#     filtered_sh = original_data.sh[mask]

#     # Fill the structured array with data from the filtered attributes
#     max_sh_degree = 3
#     data = {
#         'x': filtered_xyz[:, 0],
#         'y': filtered_xyz[:, 1],
#         'z': filtered_xyz[:, 2],
#         'nx': filtered_rot[:, 0],
#         'ny': filtered_rot[:, 1],
#         'nz': filtered_rot[:, 2],
#         'f_dc_0': filtered_sh[:, 0].flatten(),
#         'f_dc_1': filtered_sh[:, 1].flatten(),
#         'f_dc_2': filtered_sh[:, 2].flatten(),
#         **{f'f_rest_{i}': filtered_sh[:, i].flatten() for i in range(3 * ((max_sh_degree + 1) ** 2 - 1))},
#         'opacity': filtered_opacity.flatten(),
#         'scale_0': filtered_scale[:, 0],
#         'scale_1': filtered_scale[:, 1],
#         'scale_2': filtered_scale[:, 2],
#         'rot_0': filtered_rot[:, 0],
#         'rot_1': filtered_rot[:, 1],
#         'rot_2': filtered_rot[:, 2],
#         'rot_3': filtered_rot[:, 3]
#     }

#     # Define the data type for the structured array
#     dtype_3dgs = [
#         ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
#         ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
#         *[(f'f_rest_{i}', 'f4') for i in range(3 * ((max_sh_degree + 1) ** 2 - 1))],
#         ('opacity', 'f4'), ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
#         ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
#     ]

#     # Create a new structured numpy array for 3DGS format
#     structured_array = np.zeros(len(filtered_xyz), dtype=dtype_3dgs)

#     for field in dtype_3dgs:
#         field_name = field[0]
#         structured_array[field_name] = data[field_name]

#     # Save the converted data to the output file
#     try:
#         PlyData([PlyElement.describe(structured_array, 'vertex')], byte_order='=').write(path)
#         return True  # 成功导出文件
#     except Exception as e:
#         print(f"导出失败: {e}")
#         return False  # 导出失败

def export_ply(gaussian_data, output_path, enable_aabb, enable_obb, cube_min, cube_max, cube_rotation):
    # 计算旋转矩阵并过滤点
    rotation_matrix = util.convert_euler_angles_to_rotation_matrix(cube_rotation)
    points_center = np.mean(gaussian_data.xyz, axis=0)
    # 将数据转换为 DataFrame
    df = pd.DataFrame(gaussian_data.xyz, columns=['x', 'y', 'z'])
    # 计算变换后的点
    transformed_points = df.values - points_center

    # 过滤逻辑
    if enable_aabb == 0 and enable_obb == 0:
        mask = pd.Series(True, index=df.index)  # 创建全为 True 的 Series
    elif enable_obb == 1:
        transformed_points = pd.DataFrame(np.dot(np.linalg.inv(rotation_matrix), transformed_points.T).T, columns=['x', 'y', 'z'])
        mask = (transformed_points.ge(cube_min).all(axis=1)) & (transformed_points.le(cube_max).all(axis=1))
    elif enable_aabb == 1:
        cube_min_point = points_center + cube_min
        cube_max_point = points_center + cube_max
        transformed_points = pd.DataFrame(transformed_points, columns=['x', 'y', 'z'])
        mask = (transformed_points >= cube_min_point).all(axis=1) & (transformed_points <= cube_max_point).all(axis=1)
        
    # 获取原始数据的副本
    original_data = gaussian_data.get_original_state
    # 应用掩码并计算边界框
    filtered_xyz = original_data.xyz[mask]  # 使用原始数据
    bbox_values = tuple(np.concatenate([np.min(filtered_xyz, axis=0), np.max(filtered_xyz, axis=0)]).tolist()) if filtered_xyz.size > 0 else None

    # 准备转换参数
    convertargs = {
        'input': gaussian_data.path,  # 使用GaussianData中的路径
        'output': output_path,
        'target_format': "3dgs",
        'debug': False,  # 根据需要设置debug模式
        'rgb': False,  # 根据实际情况设置rgb参数
        'bbox': bbox_values,  # 设置计算出的bbox参数
        'density_filter': False, # 如果需要，设置density_filter参数
        'remove_flyers': False # 如果需要，设置remove_flyers参数
    }
    try:
        success = gsconverter(convertargs)
        return success
    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    gs = load_ply("C:\\Users\\MSI_NB\\Downloads\\viewers\\models\\train\\point_cloud\\iteration_7000\\point_cloud.ply")
    a = gs.flat()
    print(a.shape)
