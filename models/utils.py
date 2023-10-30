import os
from os.path import join
import numpy as np
import cv2
import skimage
import plyfile
import trimesh
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_sdf_to_mesh(sdf_values, voxel_origin, voxel_size, file_name, level=0.):
    if isinstance(sdf_values, torch.Tensor):
        sdf_values = sdf_values.detach().cpu().numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(sdf_values, 
                                level=level, spacing=voxel_size)

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]

    mesh = trimesh.Trimesh(mesh_points, faces)
    mesh.export(file_name)


def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    # c2w
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    # convert to w2c
    pose = np.linalg.inv(pose)

    return intrinsics, pose


def meshcleaning(file_name):

    mesh = trimesh.load(file_name, process=False, maintain_order=True)
    cc = mesh.split(only_watertight=False)    

    out_mesh = cc[0]
    bbox = out_mesh.bounds
    area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
    for c in cc:
        bbox = c.bounds
        if area < (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1]):
            area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
            out_mesh = c
    
    out_mesh.export(file_name)



def get_matrix(normal, degree=3):
    if isinstance(normal, np.ndarray):
        matrix = np.zeros((normal.shape[0], degree**2))
    elif isinstance(normal, torch.Tensor):
        matrix = torch.zeros(normal.shape[0], degree**2, device=normal.device)

    matrix[:,0] = 1
    if degree > 1:
        matrix[:,1] = normal[:,1]
        matrix[:,2] = normal[:,2]
        matrix[:,3] = normal[:,0]
    if degree > 2:
        matrix[:,4] = normal[:,0] * normal[:,1]
        matrix[:,5] = normal[:,1] * normal[:,2]
        matrix[:,6] = (2*normal[:,2]*normal[:,2]-normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])
        matrix[:,7] = normal[:,2] * normal[:,0]
        matrix[:,8] = (normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])

    return matrix

def get_radiance(coeff, normal, degree=3):
    '''
    coeff 9 or n 9
    normal n 3
    '''

    radiance = coeff[...,0]
    if degree > 1:
        radiance = radiance + coeff[...,1] * normal[:,1]
        radiance = radiance + coeff[...,2] * normal[:,2]
        radiance = radiance + coeff[...,3] * normal[:,0]
    if degree > 2:
        radiance = radiance + coeff[...,4] * normal[:,0] * normal[:,1]
        radiance = radiance + coeff[...,5] * normal[:,1] * normal[:,2]
        radiance = radiance + coeff[...,6] * (2*normal[:,2]*normal[:,2]-normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])
        radiance = radiance + coeff[...,7] * normal[:,2] * normal[:,0]
        radiance = radiance + coeff[...,8] * (normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])

    return radiance



def mynormalize(tensor, p=2, dim=1, eps=1e-12):
    denom = tensor.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(tensor).detach()
    return tensor / denom

'''
code adapted from pytorch3d https://github.com/facebookresearch/pytorch3d
'''

def get_normals(vertices, faces):
    '''
    vertices b n 3
    faces f 3
    '''
    verts_normals = torch.zeros_like(vertices)

    vertices_faces = vertices[:, faces] # b f 3 3

    verts_normals.index_add_(
        1,
        faces[:, 1],
        torch.cross(
            vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
            vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 2],
        torch.cross(
            vertices_faces[:, :, 0] - vertices_faces[:, :, 2],
            vertices_faces[:, :, 1] - vertices_faces[:, :, 2],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 0],
        torch.cross(
            vertices_faces[:, :, 1] - vertices_faces[:, :, 0],
            vertices_faces[:, :, 2] - vertices_faces[:, :, 0],
            dim=2,
        ),
    )

    verts_normals = F.normalize(verts_normals, p=2, dim=2, eps=1e-6)
    # verts_normals = mynormalize(verts_normals, p=2, dim=2, eps=1e-6)

    return verts_normals


def get_edges(verts, faces):
    V = verts.shape[0]
    f = faces.shape[0]
    device = verts.device
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

    edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
    edges, _ = edges.sort(dim=1)
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
    sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
    unique_mask = torch.ones(edges_hash.shape[0], dtype=torch.bool, device=device)
    unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
    edges = torch.stack([u // V, u % V], dim=1)

    faces_to_edges = inverse_idxs.reshape(3, f).t()

    return edges, faces_to_edges


def laplacian_cot(verts, faces):
    '''
    verts n 3
    faces f 3
    '''
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces] # f 3 3
    v0, v1, v2 = face_verts[:,0], face_verts[:,1], face_verts[:,2]

    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot = cot / 4.0

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]

    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    L = L + L.t()

    idx = faces.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas


def compute_laplacian(verts, faces):
    # first compute edges

    V = verts.shape[0]
    device = verts.device

    edges, faces_to_edges = get_edges(verts, faces)

    e0, e1 = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L


def laplacian_smoothing(verts, faces, method="uniform"):
    weights = 1.0 / verts.shape[0]

    with torch.no_grad():
        if method == "uniform":
            L = compute_laplacian(verts, faces)
        else:
            L, inv_areas = laplacian_cot(verts, faces)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas

    if method == "uniform":
        loss = L.mm(verts)
    elif method == "cot":
        loss = L.mm(verts) * norm_w - verts
    else:
        loss = (L.mm(verts) - L_sum * verts) * norm_w
    
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum()