import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from os.path import join
from glob import glob
from tqdm import tqdm
import argparse
from pyhocon import ConfigFactory
import numpy as np
import cv2
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop, SGD
import nvdiffrast.torch as dr
from models.utils import get_normals, get_radiance, get_matrix, meshcleaning, laplacian_smoothing, mynormalize
from models.sap import PSR2Mesh, DPSR, grid_interp, point_rasterize, sap_transform, sap_generate, gen_inputs
from get_data import get_dtu_data, get_nhr_data


def main(conf, scan_id):
    conf = ConfigFactory.parse_file(conf)
    data_path = conf.get_string('data_path')
    data_type = conf.get_string('data_type')
    num = conf.get_int('num')
    w = conf.get_int('w')
    h = conf.get_int('h')
    resolution = (h, w)
    sfs_weight = conf.get_float('sfs_weight')
    lap_weight = conf.get_float('lap_weight')
    albedeo_weight = conf.get_float('albedo_weight')
    mask_weight = conf.get_float('mask_weight')
    edge_weight = conf.get_float('edge_weight')
    degree = conf.get_int('degree')
    batch = conf.get_int('batch')
    input_mesh_dire = conf.get_string('input_mesh_dire')
    out_mesh_dire = conf.get_string('out_mesh_dire')
    lr = conf.get_float('lr')
    albedo_lr = conf.get_float('albedo_lr')

    mesh = trimesh.load(join(input_mesh_dire, '%d.obj'%scan_id), process=False, maintain_order=True)
    vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()
    faces = torch.from_numpy(mesh.faces.astype(np.int32)).cuda()

    glctx = dr.RasterizeGLContext()

   
    if data_type == 'nhr':
        imgs, grayimgs, masks, w2cs, projs = get_nhr_data(data_path, scan_id, num, (w,h))
    num = imgs.shape[0]

    # compute sphere harmonic coefficient as initialization
    with torch.no_grad():
        valid_normals = []
        valid_grayimgs = []
        for k in range(0,num,batch):
            n = min(num, k+batch) - k
            w2c = w2cs[k:k+batch]
            proj = projs[k:k+batch]
            mask = masks[k:k+batch]
            grayimg = grayimgs[k:k+batch]

            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:,:,:3], faces.long())
            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat, _ = dr.interpolate(normals, rast_out, faces)
            pred_normals = feat.contiguous()
            pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, faces)
            pred_normals = F.normalize(pred_normals,p=2,dim=3)
            valid_idx = (mask > 0) & (rast_out[:,:,:,3] > 0)
            valid_normals.append(pred_normals[valid_idx].detach().cpu().numpy())
            valid_grayimgs.append(grayimg[valid_idx].detach().cpu().numpy())

    valid_normals = np.concatenate(valid_normals, axis=0)
    valid_grayimgs = np.concatenate(valid_grayimgs, axis=0)

    matrix = get_matrix(valid_normals, degree)
    sh_coeff = np.linalg.lstsq(matrix, valid_grayimgs, rcond=None)[0]

    sh_coeff = torch.from_numpy(sh_coeff.astype(np.float32)).cuda()
    sh_coeff.requires_grad_(True)

    albedo = torch.ones_like(vertices).unsqueeze(0) * 0.01
    albedo.requires_grad_(True)
    optimizer = Adam([{'params': albedo, 'lr': 0.02}, {'params': sh_coeff, 'lr': 0.001}])

    pbar = tqdm(range(100))

    for i in pbar:
        perm = torch.randperm(num).cuda()
        for k in range(0, num, batch):
            n = min(num, k+batch) - k
            w2c = w2cs[perm[k:k+batch]]
            proj = projs[perm[k:k+batch]]
            img = imgs[perm[k:k+batch]]
            mask = masks[perm[k:k+batch]]

            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:,:,:3], faces.long())

            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat = torch.cat([normals, albedo.expand(n,-1,-1)], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, faces)
            pred_normals = feat[:,:,:,:3].contiguous()
            rast_albedo = feat[:,:,:,3:6].contiguous()
            pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, faces)
            pred_normals = F.normalize(pred_normals,p=2,dim=3)
            rast_albedo = dr.antialias(rast_albedo, rast_out, proj_verts, faces)

            valid_idx = (mask > 0) & (rast_out[:,:,:,3] > 0)
            valid_normals = pred_normals[valid_idx]
            valid_albedo = rast_albedo[valid_idx]

            valid_img = img[valid_idx]
            radiance = get_radiance(sh_coeff, valid_normals, degree).unsqueeze(-1)
            pred_img = radiance * valid_albedo

            sfs_loss = sfs_weight * F.l1_loss(pred_img, valid_img)
            albedo_loss = albedeo_weight * laplacian_smoothing(albedo.squeeze(0), faces.long(), method="uniform")

            loss = sfs_loss + albedo_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            des = 'sfs:%.4f'%sfs_loss.item() + ' albedo:%.4f'%albedo_loss.item()
            pbar.set_description(des)


    vertices.requires_grad_(True)
    verts_optimizer = Adam([{'params': vertices, 'lr': lr}, {'params': sh_coeff, 'lr': 0.001}])
    albedo_optimizer = Adam([{'params': albedo, 'lr': albedo_lr}])

    pbar = tqdm(range(100))
    for i in pbar:
        perm = torch.randperm(num).cuda()
        for k in range(0, num, batch):
            n = min(num, k+batch) - k
            w2c = w2cs[perm[k:k+batch]]
            proj = projs[perm[k:k+batch]]
            img = imgs[perm[k:k+batch]]
            mask = masks[perm[k:k+batch]]

            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:,:,:3], faces.long())

            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat = torch.cat([normals, albedo.expand(n,-1,-1), torch.ones_like(vertsw[:,:,:1])], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, faces)
            pred_normals = feat[:,:,:,:3].contiguous()
            rast_albedo = feat[:,:,:,3:6].contiguous()
            pred_mask = feat[:,:,:,6:7].contiguous()
            pred_normals = F.normalize(pred_normals,p=2,dim=3)
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces).squeeze(-1)

            valid_idx = (mask > 0) & (rast_out[:,:,:,3] > 0)
            valid_normals = pred_normals[valid_idx]
            valid_albedo = rast_albedo[valid_idx]

            valid_img = img[valid_idx]
            radiance = get_radiance(sh_coeff, valid_normals, degree).unsqueeze(-1)
            pred_img = radiance * valid_albedo

            tmp_img = torch.zeros_like(img)
            tmp_img[valid_idx] = pred_img
            tmp_img = dr.antialias(tmp_img, rast_out, proj_verts, faces)

            sfs_loss = sfs_weight * F.l1_loss(tmp_img[valid_idx], valid_img)

            lap_loss  = lap_weight * laplacian_smoothing(vertices, faces.long(), method="uniform")
            albedo_loss = albedeo_weight * laplacian_smoothing(albedo.squeeze(0), faces.long(), method="uniform")
            

            mask_loss = mask_weight * F.mse_loss(pred_mask, mask)

            a = vertices[faces[:, 0].long()]
            b = vertices[faces[:, 1].long()]
            c = vertices[faces[:, 2].long()]
            edge_loss = edge_weight * torch.cat([((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)]).mean()

            loss = sfs_loss + lap_loss + albedo_loss + mask_loss + edge_loss

            verts_optimizer.zero_grad()
            albedo_optimizer.zero_grad()
            # optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            verts_optimizer.step()
            albedo_optimizer.step()

            des = 'sfs:%.4f'%sfs_loss.item() + ' lap:%.4f'%lap_loss.item() + ' albedo:%.4f'%albedo_loss.item() +\
                 ' mask:%.4f'%mask_loss.item() + ' edge:%.4f'%edge_loss.item()
            pbar.set_description(des)

    torch.save({'sh_coeff': sh_coeff, 'albedo': albedo}, join(out_mesh_dire, '%d.pt'%scan_id))

    np_verts = vertices.squeeze().detach().cpu().numpy()
    np_faces = faces.squeeze().detach().cpu().numpy()

    mesh = trimesh.Trimesh(np_verts, np_faces)
    mesh.export(join(out_mesh_dire, '%d.obj'%scan_id))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/thu_sfs.conf')
    parser.add_argument('--scan_id', type=int, default=0)
    args = parser.parse_args()
    main(args.conf, args.scan_id)