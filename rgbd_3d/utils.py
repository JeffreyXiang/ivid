import time
import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
import glm
import cv2
from PIL import Image
import plyfile

# from moderngl_renderer import SimpleRenderer, AggregationRenderer


def save_ply(filename, mesh):
    vertices = edict({k: v.astype(np.float32) for k, v in mesh.vertices.items()})
    faces = mesh.faces.astype(np.int32)

    vertices_data = np.zeros(len(vertices.position), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices_data['x'] = vertices.position[:, 0]
    vertices_data['y'] = vertices.position[:, 1]
    vertices_data['z'] = vertices.position[:, 2]
    vertices_data['red'] = np.clip(vertices.color[:, 0] * 255, 0, 255).astype(np.uint8)
    vertices_data['green'] = np.clip(vertices.color[:, 1] * 255, 0, 255).astype(np.uint8)
    vertices_data['blue'] = np.clip(vertices.color[:, 2] * 255, 0, 255).astype(np.uint8)
    faces_data = np.zeros(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    faces_data['vertex_indices'] = faces

    plyfile.PlyData([
        plyfile.PlyElement.describe(vertices_data, 'vertex'),
        plyfile.PlyElement.describe(faces_data, 'face')
    ]).write(filename)


def to8b(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def linearize_depth(depth, near=0.5, far=100, mode='z_buffer'):
    """
    Linearize depth.

    Args:
        depth (np.ndarray): depth image
        near (float): near plane for perspective projection
        far (float): far plane for perspective projection
        mode (str): how the depth is stored
            - 'z_buffer': perspective projection to [0, 1]
            - 'linear': linear mapping to [0, 1]

    Returns:
        depth (np.ndarray): linearized depth image
    """
    if mode == 'z_buffer':
        depth = np.clip(depth, 1e-6, 1.0-1e-6)
        depth = near * far / (far - (far - near) * depth)
    elif mode == 'linear':
        depth = near + (far - near) * (depth)
    return depth


def project_depth(depth, near=0.5, far=100, mode='z_buffer'):
    if mode == 'z_buffer':
        depth = np.clip(depth, near, far)
        depth = (1 / near - 1 / depth) / (1 / near - 1 / far)
    elif mode == 'linear':
        depth = (depth - near) / (far - near)
    return depth


def image_uv(image_size):
    """
    Generate uv coordinates of an image.

    Args:
        image_size (int): size of the image

    Returns:
        uv (np.ndarray): uv coordinates
    """
    uv = np.meshgrid(
        np.linspace(0.5 / image_size, 1 - 0.5 / image_size, image_size),
        np.linspace(0.5 / image_size, 1 - 0.5 / image_size, image_size),
        indexing='xy'
    )
    uv = np.stack(uv, axis=-1)
    return uv


def unproject(depth, fov=45):
    """
    unproject depth to 3D points.

    Args:
        depth (np.ndarray): depth image (linearized)
        fov (float): field of view
        near (float): near plane for perspective projection
        far (float): far plane for perspective projection
        mode (str): how the depth is stored

    Returns:
        points (np.ndarray): 3D points
        uv (np.ndarray): uv coordinates
    """
    image_size = depth.shape[0]
    fov = np.deg2rad(fov)
    uv = image_uv(image_size)
    focal = 0.5 / np.tan(0.5 * fov)
    pts = np.concatenate([(uv - 0.5) / focal, -np.ones((image_size, image_size, 1))], axis=-1)
    pts = pts[::-1] * depth
    return pts, uv


def triangulate(points):
    """
    Triangulate points.

    Args:
        points (np.ndarray): points

    Returns:
        faces (np.ndarray): mesh faces
    """
    indices = np.arange(points.shape[0] * points.shape[1]).reshape(points.shape[:2])
    face_type = np.linalg.norm(points[:-1, :-1] - points[1:, 1:], axis=-1) < np.linalg.norm(points[:-1, 1:] - points[1:, :-1], axis=-1)
    faces = np.stack([
        indices[:-1, 1:].reshape(-1),
        indices[:-1, :-1].reshape(-1),
        np.where(face_type, indices[1:, 1:], indices[1:, :-1]).reshape(-1),
        indices[1:, :-1].reshape(-1),
        indices[1:, 1:].reshape(-1),
        np.where(face_type, indices[:-1, :-1], indices[:-1, 1:]).reshape(-1)
    ], axis=-1)
    faces = faces.reshape(-1, 3)
    return faces


def mask_discontinuity(faces, depths, atol=0.02, rtol=0.02):
    depths = depths.reshape(-1)
    diff = np.max(depths[faces], axis=-1) - np.min(depths[faces], axis=-1)
    inv_diff = np.max(1 / depths[faces], axis=-1) - np.min(1 / depths[faces], axis=-1)
    return np.logical_and(diff > atol, inv_diff > rtol)


def depth_to_mesh(
        depth,
        padding=None,
        fov=45,
        modelview=None,
        atol=None,
        rtol=None,
        erode_rgb=None,
        cal_normal=False,
    ):
    """
    Convert depth to mesh.
    
    Args:
        depth (np.ndarray): depth image (linearized)
        padding (float): padding for the depth image
        fov (float): field of view
        near (float): near plane for perspective projection
        far (float): far plane for perspective projection
        mode (str): how the depth is stored
        modelview (glm.mat4): modelview matrix
        atol (float): absolute tolerance for depth difference
        rtol (float): relative tolerance for depth difference

    Returns:
        vertices (np.ndarray): mesh vertices
        faces (np.ndarray): mesh faces
    """
    image_size = depth.shape[0]
    image_plane_size = 2 * np.tan(0.5 * np.deg2rad(fov))
    points, uv = unproject(depth, fov)
    if cal_normal:
        normal = cal_depth_normal(points)

    ret = edict({
        'depth': depth,
        'fov': fov,
        'modelview': modelview,
    })

    if padding is not None:
        points = np.pad(points, ((1, 1), (1, 1), (0, 0)), 'edge')
        uv = np.pad(uv, ((1, 1), (1, 1), (0, 0)), 'edge')
        depth = np.pad(depth, ((1, 1), (1, 1), (0, 0)), 'edge')
        if cal_normal:
            normal = np.pad(normal, ((1, 1), (1, 1), (0, 0)), 'edge')
        if padding == 'frustum':
            padding_per_pixel = image_plane_size / image_size
            points[0, :, 1] += padding_per_pixel * depth[0, :, 0]
            points[-1, :, 1] -= padding_per_pixel * depth[-1, :, 0]
            points[:, 0, 0] -= padding_per_pixel * depth[:, 0, 0]
            points[:, -1, 0] += padding_per_pixel * depth[:, -1, 0]
            points[0, :] *= -0.1 / points[0, :, 2:]
            points[-1, :] *= -0.1 / points[-1, :, 2:]
            points[:, 0] *= -0.1 / points[:, 0, 2:]
            points[:, -1] *= -0.1 / points[:, -1, 2:]
        else:
            padding_per_pixel = padding * image_plane_size / image_size
            points[0, :, 1] += padding_per_pixel * depth[0, :, 0]
            points[-1, :, 1] -= padding_per_pixel * depth[-1, :, 0]
            points[:, 0, 0] -= padding_per_pixel * depth[:, 0, 0]
            points[:, -1, 0] += padding_per_pixel * depth[:, -1, 0]
        padding_flag = np.zeros_like(depth, dtype=np.bool_)
        padding_flag[0, :] = True
        padding_flag[-1, :] = True
        padding_flag[:, 0] = True
        padding_flag[:, -1] = True
        image_size += 2
    else:
        padding_flag = np.zeros_like(depth, dtype=np.bool_)

    faces = triangulate(points)

    points = points.reshape(-1, 3)
    if cal_normal:
        normal = normal.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    uv = uv.reshape(-1, 2)
    depth = depth.reshape(-1, 1)
    padding_flag = padding_flag.reshape(-1, 1)

    discontinuity_flag = np.zeros_like(depth, dtype=np.bool_)
    if atol is not None or rtol is not None:
        atol = 0 if atol is None else atol
        rtol = 0 if rtol is None else rtol
        mask = mask_discontinuity(faces, depth, atol=atol, rtol=rtol)
        discontinuity_flag[faces[mask, :]] = True

    if modelview is not None:
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        points = np.matmul(glm.inverse(modelview), points.T).T
        points = points[:, :3]
        if cal_normal:
            normal = np.matmul(glm.mat3(glm.inverse(modelview)), normal.T).T

    erosion_flag = np.zeros_like(depth, dtype=np.bool_)
    if erode_rgb is not None and erode_rgb > 0:
        mask = np.ones_like(discontinuity_flag, dtype=np.float32)
        mask[discontinuity_flag] = 0
        mask = mask.reshape(image_size, image_size)
        erode_radius = 2 * erode_rgb + 1
        mask = cv2.erode(mask, np.ones((erode_radius, erode_radius)))
        mask = mask.reshape(-1, 1)
        erosion_flag[mask == 0] = True

    flag = 1 * discontinuity_flag + 2 * padding_flag + 4 * erosion_flag

    ret['faces'] = faces
    ret['vertices'] = edict({
        'position': points,
        'uv': uv,
        'flag': flag,
    })
    if cal_normal:
        ret.vertices['normal'] = normal

    return ret


def cal_depth_normal(points):
    """
    Calculate mesh normal with sobel filter.
    """
    points = np.pad(points, ((1, 1), (1, 1), (0, 0)), 'edge')
    edge_x = points[:, 2:] - points[:, :-2]
    edge_y = points[:-2, :] - points[2:, :] 
    edge_x = (1 * edge_x[:-2, :] + 2 * edge_x[1:-1, :] + 1 * edge_x[2:, :]) / 4
    edge_y = (1 * edge_y[:, :-2] + 2 * edge_y[:, 1:-1] + 1 * edge_y[:, 2:]) / 4
    normal = np.cross(edge_x, edge_y)
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal


def cal_mesh_normal(vertices, faces):
    """
    Calculate mesh normal.

    Args:
        vertices (np.ndarray): mesh vertices
        faces (np.ndarray): mesh faces
    """
    ## using weighted average of face normals
    points = vertices[:, :3]
    edge0 = points[faces[:, 1]] - points[faces[:, 0]]
    edge1 = points[faces[:, 2]] - points[faces[:, 1]]
    edge2 = points[faces[:, 0]] - points[faces[:, 2]]
    edge0 = edge0 / np.linalg.norm(edge0, axis=-1, keepdims=True)
    edge1 = edge1 / np.linalg.norm(edge1, axis=-1, keepdims=True)
    edge2 = edge2 / np.linalg.norm(edge2, axis=-1, keepdims=True)
    face_normals = np.cross(edge0, -edge2)
    face_normals = face_normals / np.linalg.norm(face_normals, axis=-1, keepdims=True)
    face_angles = np.arccos(np.stack([
        np.sum(-edge0 * edge2, axis=-1),
        np.sum(-edge0 * edge1, axis=-1),
        np.sum(-edge1 * edge2, axis=-1)
    ], axis=-1))
    normals = np.zeros((vertices.shape[0], 3))
    for i in range(3):
        normals[:, 0] += np.bincount(faces[:, i], weights=face_normals[:, 0] * face_angles[:, i], minlength=normals.shape[0])
        normals[:, 1] += np.bincount(faces[:, i], weights=face_normals[:, 1] * face_angles[:, i], minlength=normals.shape[0])
        normals[:, 2] += np.bincount(faces[:, i], weights=face_normals[:, 2] * face_angles[:, i], minlength=normals.shape[0])
        # np.add.at(normals, faces[:, i], face_normals * face_angles[:, i:i+1])
        # np.add.at(normals_weight, faces[:, i], face_angles[:, i])
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals


def depth_edge(depth, atol=0.02, rtol=0.02):
    def depth_diff(x, y):
        x = np.maximum(x, 1e-6)
        y = np.maximum(y, 1e-6)
        diff = np.abs(x - y)
        inv_diff = np.abs(1 / x - 1 / y)
        return np.logical_and(diff > atol, inv_diff > rtol)
    # mask out depth edge
    mask = np.zeros((depth.shape[0], depth.shape[1], 1), dtype=np.uint8)
    mask_ = depth_diff(depth[:, 1:], depth[:, :-1])
    mask[:, 1:] += mask_
    mask[:, :-1] += mask_
    mask_ = depth_diff(depth[1:, :], depth[:-1, :])
    mask[1:, :] += mask_
    mask[:-1, :] += mask_
    mask_ = depth_diff(depth[1:, 1:], depth[:-1, :-1])
    mask[1:, 1:] += mask_
    mask[:-1, :-1] += mask_
    mask_ = depth_diff(depth[1:, :-1], depth[:-1, 1:])
    mask[1:, :-1] += mask_
    mask[:-1, 1:] += mask_
    return mask < 3


def forward_backward_warp(
        renderer,
        rgbd,
        modelview1,
        modelview0=None,
        padding=None,
        fov=45,
        near=0.5,
        far=100,
        mode='z_buffer',
        atol=0.02,
        rtol=0.02
    ):
    """
    Warp back and forth between two views.

    Args:
        renderer (OpenglRenderer): renderer
        rgbd (np.ndarray): RGBD image
        modelview1 (glm.mat4): modelview matrix of view1
        modelview0 (glm.mat4): modelview matrix of view0
        padding (float): padding for the depth image
        fov (float): field of view
        near (float): near plane for perspective projection
        far (float): far plane for perspective projection
        mode (str): how the depth is stored
        atol (float): absolute tolerance for depth difference
        rtol (float): relative tolerance for depth difference
    """
    image_size = rgbd.shape[0]
    ssaa = renderer.render_size // image_size
    ssaa_offset = (ssaa - 1) // 2

    # backproject view0
    if modelview0 is None:
        modelview0 = glm.lookAt(
            glm.vec3(0.0, 0.0, 1.0),
            glm.vec3(0.0, 0.0, 0.0),
            glm.vec3(0.0, 1.0, 0.0)
        )
    mesh0 = depth_to_mesh(
        linearize_depth(rgbd[:, :, 3:], near, far, mode),
        padding=padding,
        fov=fov,
        modelview=modelview0,
        atol=None,
        rtol=None,
    )

    # render from view1
    res = renderer.render(mesh0, rgbd[:, :, :3], modelview1, fov)
    color1 = np.array(Image.fromarray(to8b(res.color)).resize((image_size, image_size), Image.Resampling.LANCZOS)) / 255.0
    depth1 = res.depth[ssaa_offset::ssaa, ssaa_offset::ssaa, :]

    # backproject view1
    mesh1 = depth_to_mesh(
        depth1,
        padding=None,
        fov=fov,
        modelview=modelview1,
        atol=atol,
        rtol=rtol,
    )

    # render from view0
    res = renderer.render(mesh1, color1, modelview0, fov)
    color = np.array(Image.fromarray(to8b(res.color)).resize((image_size, image_size), Image.Resampling.LANCZOS)) / 255.0
    depth = res.depth[ssaa_offset::ssaa, ssaa_offset::ssaa, :]
    depth = project_depth(depth, near, far, mode)
    mask = res.mask.reshape(image_size, ssaa, image_size, ssaa, 1).sum(axis=(1, 3)) > 0.75 * ssaa**2

    # mask out depth edge
    mask &= depth_edge(depth, atol=atol, rtol=rtol)
    
    color *= mask
    depth *= mask
    mask = mask.astype(np.float32)

    return edict({
        'color': color,
        'depth': depth,
        'mask': mask,
    })


def aggregate_conditions(
        renderer,
        meshes,
        colors,
        modelview,
        fov=45,
        near=0.5,
        mode='z_buffer',
        far=100,
        atol=0.02,
        rtol=0.02,
        erode_rgb=2,
    ):
    """
    Aggregate conditions from multiple views.

    Args:
        renderer (AggregationRenderer): renderer
        meshes (list): list of meshes
        colors (list): list of colors
        modelview (glm.mat4): modelview matrix
        fov (float): field of view
        near (float): near plane for perspective projection
        far (float): far plane for perspective projection
        mode (str): how the depth is stored
        atol (float): absolute tolerance for depth difference
        rtol (float): relative tolerance for depth difference
        erode_rgb (int): radius of erosion for rgb
    """
    image_size = colors[0].shape[0]
    ssaa = renderer.render_size // image_size
    ssaa_offset = (ssaa - 1) // 2

    res = renderer.render(meshes, colors, modelview, fov, is_autoregressive=True)
    color = np.array(Image.fromarray(to8b(res.color)).resize((image_size, image_size), Image.Resampling.LANCZOS)) / 255.0
    depth = res.depth[ssaa_offset::ssaa, ssaa_offset::ssaa, :]
    depth = project_depth(depth, near, far, mode)
    mask = res.mask_depth.reshape(image_size, ssaa, image_size, ssaa, 1).sum(axis=(1, 3)) > 0.75 * ssaa**2
    mask_rgb = res.mask_color.reshape(image_size, ssaa, image_size, ssaa, 1).sum(axis=(1, 3)) > 0.75 * ssaa**2
    depth_convex = depth.copy()

    # mask out depth edge
    edge = depth_edge(depth, atol=atol, rtol=rtol)
    mask &= edge
    mask_rgb &= cv2.erode(mask.astype(np.uint8)[..., 0], np.ones((2 * erode_rgb - 1, 2 * erode_rgb - 1), np.uint8), iterations=1)[..., None] > 0

    color *= mask_rgb
    depth *= mask
    mask = mask.astype(np.float32)
    mask_rgb = mask_rgb.astype(np.float32)

    return edict({
        'color': color,
        'depth': depth,
        'mask': mask,
        'mask_rgb': mask_rgb,
        'depth_convex': depth_convex,
    })
