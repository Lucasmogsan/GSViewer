import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
import scipy as sp
import open3d as o3d

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
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
        center = np.mean(self.xyz, axis=0)
        centered_points = self.xyz - center
        covariance_matrix = np.cov(centered_points, rowvar=False)
        U, S, Vt = np.linalg.svd(covariance_matrix)
        projected_points = centered_points @ U
        min_bounds = np.min(projected_points, axis=0)
        max_bounds = np.max(projected_points, axis=0)
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
        return obb_min, obb_max, U, obb_corners

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
#     return GaussianData(xyz, rots, scales, opacities, shs)

def load_ply(path, scale_to_interval):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    # 计算模型的边界
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    center_xyz = (min_xyz + max_xyz) / 2
    max_extent = np.max(max_xyz - min_xyz)
    # 计算缩放因子
    scale_factor = scale_to_interval / max_extent
    # 应用缩放和中心化
    xyz = (xyz - center_xyz) * scale_factor

    # 处理法线
    normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])), axis=1)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # 处理其他属性
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    scales = np.exp(scales) * scale_factor  # 调整局部缩放

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)

    opacities = 1 / (1 + np.exp(-opacities))  # sigmoid
    shs = np.concatenate([features_dc.reshape(-1, 3), features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)

    return GaussianData(xyz.astype(np.float32), rots.astype(np.float32), scales.astype(np.float32), opacities.astype(np.float32), shs.astype(np.float32))

if __name__ == "__main__":
    gs = load_ply("C:\\Users\\MSI_NB\\Downloads\\viewers\\models\\train\\point_cloud\\iteration_7000\\point_cloud.ply")
    a = gs.flat()
    print(a.shape)
