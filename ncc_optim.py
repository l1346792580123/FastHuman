import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from os.path import join
from glob import glob
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import trimesh
from pyhocon import ConfigFactory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop, SGD
import nvdiffrast.torch as dr
from models.utils import get_normals, meshcleaning
from models.ncc_utils import build_patch_offset, NCC
from models.sap import PSR2Mesh, DPSR, sap_transform, sap_generate, gen_inputs
from get_data import get_dtu_data, get_nhr_data


def main(conf, scan_id):
    conf = ConfigFactory.parse_file(conf)
    data_path = conf.get_string('data_path')
    data_type = conf.get_string('data_type')
    num = conf.get_int('num')
    w = conf.get_int('w')
    h = conf.get_int('h')
    sap_res = conf.get_int('sap_res')
    sig = conf.get_int('sig')
    num_points = conf.get_int('num_points')
    num_sample = conf.get_int('num_sample')
    h_patch_size = conf.get_int('h_patch_size')
    ncc_thresh = conf.get_float('ncc_thresh')
    lr = conf.get_float('lr')
    rgb_ncc = conf.get_bool('rgb_ncc')
    ncc_weight = conf.get_float('ncc_weight')
    mask_weight = conf.get_float('mask_weight')
    pair_file = conf.get_string('pair_file')
    input_mesh_dire = conf.get_string('input_mesh_dire')
    out_mesh_dire = conf.get_string('out_mesh_dire')
    need_mask = conf.get_bool('need_mask')
    sparse_file = conf.get_string('sparse_file')
    atol = conf.get_float('atol')
    align_corners = False

    if sparse_file != 'none':   
        with open(('dtu_sparse/%d.pkl'%scan_id), 'rb') as f:
            img_points = pickle.load(f, encoding='latain1')
        use_sparse = True
    else:
        use_sparse = False

    resolution = (h, w)
    num_pixels = (h_patch_size*2+1)**2

    psr2mesh = PSR2Mesh.apply
    dpsr = DPSR((sap_res,sap_res,sap_res), sig).cuda()
    glctx = dr.RasterizeGLContext()

    if data_type == 'dtu':
        imgs, grayimgs, masks, w2cs, projs = get_dtu_data(data_path, scan_id, (w,h))
    elif data_type == 'nhr':
        imgs, grayimgs, masks, w2cs, projs = get_nhr_data(data_path, scan_id, num, (w,h))

    poses = w2cs.permute(0,2,1).contiguous()
    num = imgs.shape[0]

    with open(pair_file) as f:
        pp = f.readlines()
    fun = lambda s: int(s)
    pairs = []
    for p in pp:
        splitted = p.split()[1:]  # drop the first one since it is the ref img
        pairs.append(torch.tensor(list(map(fun, splitted))).cuda())

    offsets = build_patch_offset(h_patch_size, pairs[0].device).float()

    inputs, center, scale = gen_inputs(join(input_mesh_dire, '%d.obj'%scan_id), num_sample)
    inputs = inputs.cuda()
    center = center.cuda()
    scale = scale.cuda()
    inputs.requires_grad_(True)
    if need_mask:
        inputs_optimizer = Adam([{'params': inputs, 'lr': 0.01}])
        batch = 8
        dpsr_s = DPSR((128,128,128), sig).cuda()
        pbar = tqdm(range(201))
        for i in pbar:
            perm = torch.randperm(num).cuda()
            for k in range(0, num,8):
                n = min(num, k+batch) - k
                w2c = w2cs[perm[k:k+batch]]
                proj = projs[perm[k:k+batch]]
                mask = masks[perm[k:k+batch]]

                vertices, faces, v, psr_grid, points = sap_generate(dpsr_s, psr2mesh, inputs, center, scale)
                vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
                rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
                proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)

                rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
                feat =  feat = torch.cat([torch.ones_like(vertsw[:,:,:1]), vertsw[:,:,:3]], dim=2)
                feat, _ = dr.interpolate(feat, rast_out, faces)
                pred_mask = feat[:,:,:,:1].contiguous()
                rast_points = feat[:,:,:,1:4].contiguous()
                pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces).squeeze(-1)
                rast_points = dr.antialias(rast_points, rast_out, proj_verts, faces)

                mask_loss = mask_weight * F.mse_loss(pred_mask, mask)

                if use_sparse:
                    sparse_points = torch.from_numpy(img_points[perm[k].item()]['points'].astype(np.float32)).cuda()
                    sparse_uvs = torch.from_numpy(img_points[perm[k].item()]['uvs'].astype(np.float32)).cuda()
                    sparse_uvs[:,0] = (sparse_uvs[:,0] - (w/2)) / (w/2)
                    sparse_uvs[:,1] = (sparse_uvs[:,1] - (h/2)) / (h/2)
                    sparse_uvs = sparse_uvs.reshape(1,-1,1,2)
                    sampled_points = F.grid_sample(rast_points[0:1].permute(0,3,1,2).contiguous(), sparse_uvs, align_corners=align_corners).squeeze()
                    sparse_loss = 10 * F.l1_loss(sparse_points, sampled_points.permute(1,0).contiguous())
                else:
                    sparse_loss = torch.zeros_like(mask_loss)

                total_loss = mask_loss + sparse_loss

                inputs_optimizer.zero_grad()
                total_loss.backward()
                inputs_optimizer.step()

                des = ' m:%.4f'%mask_loss.item()
                pbar.set_description(des)

            if i % 10 == 0 and i != 0:
                with torch.no_grad():
                    vertices, faces, v, psr_grid, points = sap_generate(dpsr_s, psr2mesh, inputs, center, scale)

                    save_verts = vertices.squeeze(0).detach().cpu().numpy()
                    np_faces = faces.squeeze(0).detach().cpu().long().numpy()
                    save_mesh = trimesh.Trimesh(save_verts, np_faces, process=False, maintain_order=True)
                    save_mesh.export(join(out_mesh_dire, '%d.obj'%scan_id))

                    inputs, center, scale = gen_inputs(join(out_mesh_dire, '%d.obj'%scan_id), num_sample)
                    inputs = inputs.cuda()
                    center = center.cuda()
                    scale = scale.cuda()
                    inputs.requires_grad_(True)

                    del inputs_optimizer
                    inputs_optimizer = Adam([{'params': inputs, 'lr': 0.01}])
    else:
        inputs_optimizer = Adam([{'params': inputs, 'lr': lr}])

    optim_epoch = 10
    pbar = tqdm(range(optim_epoch))
    # torch.autograd.set_detect_anomaly(True)
    for i in pbar:
        perm = torch.randperm(num).cuda()
        for k in range(0, num):
            ref_w2c = w2cs[perm[k:k+1]]
            ref_pose = poses[perm[k:k+1]]
            ref_proj = projs[perm[k:k+1]]
            ref_gray = grayimgs[perm[k:k+1]]
            ref_img = imgs[perm[k]]
            ref_mask = masks[perm[k:k+1]]
            src_w2c = w2cs[pairs[perm[k]]]
            src_pose = poses[pairs[perm[k]]]
            src_proj = projs[pairs[perm[k]]]
            src_gray = grayimgs[pairs[perm[k]]]
            src_img = imgs[pairs[perm[k]]]
            src_mask = masks[pairs[perm[k]]]

            rel_pose = src_pose @ torch.inverse(ref_pose)

            w2c = torch.cat([ref_w2c, src_w2c])
            proj = torch.cat([ref_proj, src_proj])
            mask = torch.cat([ref_mask, src_mask])
            n = w2c.shape[0]

            vertices, faces, v, psr_grid, points = sap_generate(dpsr, psr2mesh, inputs, center, scale)
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)

            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat = torch.cat([rot_verts[:,:,:3], torch.ones_like(vertsw[:,:,:1]), vertsw[:,:,:3]], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, faces)
            rast_verts = feat[:,:,:,:3].contiguous()
            pred_mask = feat[:,:,:,3:4].contiguous()
            rast_points = feat[:,:,:,4:7].contiguous()
            # antialias may change surrounding zero pixel to non zero
            # rast_verts = dr.antialias(rast_verts, rast_out, proj_verts, faces) 
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces).squeeze(-1)
            # rast_points = dr.antialias(rast_points, rast_out, proj_verts, faces)

            # NCC compute
            valid_mask = (rast_out[0,:,:,3] > 0) & (ref_mask[0] > 0)
            ref_valid_idx = torch.where(valid_mask)
            rand_idx = torch.randperm(len(ref_valid_idx[0]))
            ref_idx = [item[rand_idx][:num_points] for item in ref_valid_idx] # part sample
            uv = torch.stack([ref_idx[1], ref_idx[0]], dim=1).unsqueeze(1) # npoints 1 2
            npoints = uv.shape[0]
            pixels = (uv + offsets).reshape(-1,2) # npoints*npixels 2
            uu = torch.clamp(pixels[:,0], 0, w-1).long()
            vv = torch.clamp(pixels[:,1], 0, h-1).long()
            uv_mask = ((pixels[:,0] >= 0) & (pixels[:,0] < w) & (pixels[:,1] >= 0) & (pixels[:,1] < h)).reshape(1, npoints, num_pixels)
            ref_verts = rast_verts[0][vv, uu]
            ref_points = rast_points[0][vv,uu]
            sampled_ref_gray = ref_gray[:, vv, uu].reshape(1, npoints, num_pixels)
            sampled_ref_img = ref_img[vv,uu].reshape(npoints, num_pixels, 3).permute(2,0,1).contiguous()
            ref_valid_mask = valid_mask[vv,uu].reshape(1, npoints, num_pixels) & uv_mask

            src_verts = (src_pose[:,:3,:3]@ref_points.permute(1,0).contiguous() + src_pose[:,:3,3:4]).permute(0,2,1).contiguous()
            # tmp = (rel_pose[:,:3,:3]@ref_verts.permute(1,0).contiguous() + rel_pose[:,:3,3:4]).permute(0,2,1).contiguous()
            # (tmp[ref_valid_mask.reshape(1,-1).expand(n-1,-1)]-src_verts[ref_valid_mask.reshape(1,-1).expand(n-1,-1)]).abs().max()
            src_depth = src_verts[:,:,2].reshape(n-1, npoints, num_pixels)
            src_f = torch.stack([src_proj[:,0,0], src_proj[:,1,1]], dim=1).unsqueeze(1)
            src_c = torch.stack([src_proj[:,2,0], src_proj[:,2,1]], dim=1).unsqueeze(1)
            grid = (src_verts[:,:,:2] / (src_verts[:,:,2:3]+1e-8)) * src_f + src_c

            sampled_src_gray = F.grid_sample(src_gray.unsqueeze(1), grid.view(n-1, -1, 1, 2), align_corners=align_corners).squeeze()
            sampled_src_gray = sampled_src_gray.reshape(n-1, npoints, num_pixels)

            sampled_src_depth = F.grid_sample(rast_verts[1:,:,:,2:3].permute(0,3,1,2).contiguous(), grid.view(n-1, -1, 1, 2), align_corners=align_corners).squeeze()
            sampled_src_depth = sampled_src_depth.reshape(n-1, npoints, num_pixels)

            sampled_src_mask = F.grid_sample(src_mask.unsqueeze(-1).permute(0,3,1,2).contiguous(), grid.view(n-1, -1, 1, 2), align_corners=align_corners).squeeze()
            sampled_src_mask = sampled_src_mask.reshape(n-1, npoints, num_pixels)

            src_valid_mask = ref_valid_mask & torch.isclose(sampled_src_depth, src_depth, atol=atol) & (sampled_src_mask>0)

            sampled_src_img = F.grid_sample(src_img.permute(0,3,1,2).contiguous(), grid.view(n-1, -1, 1, 2), align_corners=align_corners).squeeze()
            sampled_src_img = sampled_src_img.reshape(n-1, 3, npoints, num_pixels)

            if rgb_ncc:
                ncc_values = 0
                for j in range(3):
                    ncc_values = ncc_values + NCC(sampled_ref_img[j:j+1], sampled_src_img[:,j], ref_valid_mask, src_valid_mask)
                ncc_values = ncc_values / 3
            else:
                ncc_values = NCC(sampled_ref_gray, sampled_src_gray, ref_valid_mask, src_valid_mask) # nview npoints

            ncc_mask = (ncc_values > ncc_thresh) & (src_valid_mask.sum(2) > num_pixels*0.75)

            # assert (ncc_values[ncc_mask]<1).all()
            ncc_values = torch.clamp(ncc_values,max=1.0)


            ncc_loss = ncc_weight * torch.sum((torch.ones_like(ncc_values)-ncc_values)*ncc_mask) / ncc_mask.sum()
            mask_loss = mask_weight * F.mse_loss(pred_mask, mask)

            if use_sparse:
                sparse_points = torch.from_numpy(img_points[perm[k].item()]['points'].astype(np.float32)).cuda()
                sparse_uvs = torch.from_numpy(img_points[perm[k].item()]['uvs'].astype(np.float32)).cuda()
                sparse_uvs[:,0] = (sparse_uvs[:,0] - (w/2)) / (w/2)
                sparse_uvs[:,1] = (sparse_uvs[:,1] - (h/2)) / (h/2)
                sparse_uvs = sparse_uvs.reshape(1,-1,1,2)
                sampled_points = F.grid_sample(rast_points[0:1].permute(0,3,1,2).contiguous(), sparse_uvs, align_corners=align_corners).squeeze()
                sparse_loss = 10 * F.l1_loss(sparse_points, sampled_points.permute(1,0).contiguous())
            else:
                sparse_loss = torch.zeros_like(mask_loss)

            total_loss = ncc_loss + mask_loss + sparse_loss

            inputs_optimizer.zero_grad()
            total_loss.backward()
            inputs_optimizer.step()

            des = 'ncc:%.4f'%ncc_loss.item() + ' m:%.4f'%mask_loss.item() + ' sp:%.4f'%sparse_loss.item()
            pbar.set_description(des)

        if i % 2 == 0 and i != 0:
            with torch.no_grad():
                vertices, faces, v, psr_grid, points = sap_generate(dpsr, psr2mesh, inputs, center, scale)

                save_verts = vertices.squeeze(0).detach().cpu().numpy()
                np_faces = faces.squeeze(0).detach().cpu().long().numpy()
                save_mesh = trimesh.Trimesh(save_verts, np_faces, process=False, maintain_order=True)
                if sap_res == 256:
                    save_mesh.subdivide()
                save_mesh.export(join(out_mesh_dire, '%d.obj'%scan_id))

                inputs, center, scale = gen_inputs(join(out_mesh_dire, '%d.obj'%scan_id), num_sample)
                inputs = inputs.cuda()
                inputs.requires_grad_(True)
                center = center.cuda()
                scale = scale.cuda()

                del inputs_optimizer
                inputs_optimizer = Adam([{'params': inputs, 'lr': lr}])

            if i == 6:
                dpsr = DPSR((sap_res,sap_res,sap_res), 2).cuda()

    f.close()

    with torch.no_grad():
        vertices, faces, v, psr_grid, points = sap_generate(dpsr, psr2mesh, inputs, center, scale)
        save_verts = vertices.squeeze(0).detach().cpu().numpy()
        np_faces = faces.squeeze(0).detach().cpu().long().numpy()
        save_mesh = trimesh.Trimesh(save_verts, np_faces, process=False, maintain_order=True)
        if sap_res == 256:
            save_mesh.subdivide()
        save_mesh.export(join(out_mesh_dire, '%d.obj'%scan_id))
        meshcleaning(join(out_mesh_dire, '%d.obj'%scan_id))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/nhr_ncc.conf')
    parser.add_argument('--scan_id', type=int, default=0)
    args = parser.parse_args()
    main(args.conf, args.scan_id)