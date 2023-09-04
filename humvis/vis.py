import cv2
import numpy as np
import copy
import colorsys
import pytorch3d
import pytorch3d.renderer
from scipy.spatial.transform import Rotation
import torch

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)

def random_colors(num, rgb=False, maximum=255):
    import random
    idxs = random.choices(list(range(0, len(_COLORS))), k=num)
    rets = []
    for idx in idxs:
        ret = _COLORS[idx] * maximum
        if not rgb:
            ret = ret[::-1]
        ret = [int(x) for x in ret]
        rets.append(ret)
    return rets

link_pairs2 = [
        [15, 13], [13, 11], [11, 5], 
        [12, 14], [14, 16], [12, 6], 
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        [9, 7], [7,5], [5, 6], [6, 8], [8, 10],
        ]

cs = random_colors(50)
mesh_colors = [colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for _ in range(50)]

def draw_results(img, poses, masks, parts, meshes, mesh_locs, face, focal=5000, instance_ids=None):
    num_instances = poses.shape[0]
    assert num_instances == len(instance_ids)

    pose_img = copy.deepcopy(img)
    mask_img = copy.deepcopy(img)
    part_img = copy.deepcopy(img)
    mask_colormap = np.zeros_like(img)
    part_colormap = np.zeros_like(img)
    for i in range(num_instances):
        # draw pose
        kpt = poses[i]
        for j in range(kpt.shape[0]):
            k = kpt[j]
            if k[2] < 0.05: continue
            cv2.circle(pose_img, (int(k[0]), int(k[1])), 3, cs[instance_ids[i]], -1)
        
        for line in link_pairs2:
            u, v = line
            xu, yu, cu = int(kpt[u][0]), int(kpt[u][1]), kpt[u][2]
            xv, yv, cv = int(kpt[v][0]), int(kpt[v][1]), kpt[v][2]
            if cu < 0.05 or cv < 0.05: continue
            cv2.line(pose_img, (xu, yu), (xv, yv), cs[instance_ids[i]], int(kpt[u][2]+kpt[v][2]) + 1)

        # draw mask
        ii = masks.shape[0] - i - 1
        nonzero = np.where(masks[ii] > 0)
        c = cs[instance_ids[ii]]
        mask_colormap[nonzero] = c

        # draw part
        ii = parts.shape[0] - i - 1
        for jj in range(20):
            if jj == 0: continue
            nonzero = np.where(parts[ii] == jj)
            c = cs[(instance_ids[ii]+jj)%50]
            part_colormap[nonzero] = c
    
    mask_img = (
        mask_img.astype(np.float32)
    )
    mask_data = cv2.addWeighted(
        mask_img.astype(np.uint8),
        1.0,
        mask_colormap,
        1.0,
        0,
    )

    part_img = (
        part_img.astype(np.float32)
    )
    part_data = cv2.addWeighted(
        part_img.astype(np.uint8),
        1.0,
        part_colormap,
        1.0,
        0,
    )

    # drawa meshes
    assert num_instances == meshes.shape[0]
    height, width = img.shape[:2]

    focals, princpts, colors = [], [], []
    for i in range(num_instances):
        focals.append(focal)
        princpt = (int(mesh_locs[i][0]), int(mesh_locs[i][1]))
        princpts.append(princpt)
        if instance_ids:
            colors.append(mesh_colors[instance_ids[i]])
        else:
            colors.append(mesh_colors[i])

    color_batch = render_mesh(meshes, face, focals, princpts, height, width, colors)
    valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
    image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
    image_vis_batch = (image_vis_batch * 255).cpu().numpy()
    
    alpha = 0.9
    mesh_img = copy.deepcopy(img)
    for i in range(num_instances):
        ii = num_instances - i - 1
        color = image_vis_batch[ii]
        valid_mask = valid_mask_batch[ii].cpu().numpy()
        mesh_img = alpha * color[:, :, :3] * valid_mask + (1 - alpha) * mesh_img * valid_mask + (1 - valid_mask) * mesh_img

    mesh_img = mesh_img.astype(np.uint8)

    return pose_img, mask_data, part_data, mesh_img

def render_mesh(vertices, faces, focal, princpt, height, width, colors, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    bs = vertices.shape[0]

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
    faces = faces.expand(bs, *faces.shape).to(device)

    vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)

    # Initialize each vertex to be white in color.
    verts_rgb = torch.tensor(colors, dtype=vertices.dtype, device=vertices.device)[:, None, :].expand_as(vertices)
    assert verts_rgb.shape[0] == bs
    # verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=focal,
        principal_point=princpt,
        in_ndc=False,
        image_size=[(height, width) for _ in range(len(focal))],
        device=device,
    )

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),   # (H, W)
        # image_size=height,   # (H, W)
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs