import numpy as np
import cv2
from glob import glob
import os
from os.path import join
from tqdm import tqdm
import trimesh
from pyhocon import ConfigFactory
from models.utils import convert_sdf_to_mesh, load_K_Rt_from_P
from get_data import campose_to_extrinsic, read_intrinsics
import argparse

def main(conf, scan_id):

    conf = ConfigFactory.parse_file(conf)
    data_path = conf.get_string('data_path')
    data_type = conf.get_string('data_type')
    num = conf.get_int('num')
    out_mesh_dire = conf.get_string('out_mesh_dire')

    silhouette = []
    projs = []
    w2cs = []

    N = 128

    if data_type == 'dtu':
        camera_dict = np.load(join(data_path, 'scan%d/imfunc4/cameras_hd.npz'%scan_id))
        imgH, imgW = 1200, 1600
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(num)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(num)]

        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            proj, w2c = load_K_Rt_from_P(P[:3, :4])

            projs.append(proj)
            w2cs.append(w2c)

        for i in range(num):
            mask = cv2.imread(join(data_path, 'scan%d/imfunc4/pmask/%03d.png'%(scan_id,i)))
            mask = mask[:,:,0] > 0
            silhouette.append(mask)

        voxel_origin = [-1.3, -1.3, -1.3]
        x_size = 2.6 / (N - 1)
        y_size = 2.6 / (N - 1)
        z_size = 2.6 / (N - 1)

    elif data_type == 'nhr':
        imgH,imgW = 1024, 1224
        c2ws = campose_to_extrinsic(np.loadtxt(join(data_path,'CamPose.inf')))
        all_w2cs = np.linalg.inv(c2ws)
        all_projs = read_intrinsics(join(data_path, 'Intrinsic.inf'))

        for i in tqdm(range(num)):
            img = cv2.imread('%s/img/%d/img_%04d.jpg'%(data_path,scan_id,i))
            if img.shape[0] == 1024 and img.shape[1] == 1224:
                mask = cv2.imread('%s/img/%d/mask/img_%04d.jpg'%(data_path,scan_id,i))[:,:,0]
                mask = (mask>127.5).astype(np.float32)
                silhouette.append(mask)
                w2cs.append(all_w2cs[i])
                proj = np.zeros([4,4])
                proj[:3,:3] = all_projs[i]
                projs.append(proj)

        pcd = np.load('%s/pointclouds/frame%d.npy'%(data_path,scan_id+1))
        xmin = pcd[:,0].min() - 0.1
        ymin = pcd[:,1].min() - 0.1
        zmin = pcd[:,2].min() - 0.1
        xmax = pcd[:,0].max() + 0.1
        ymax = pcd[:,1].max() + 0.1
        zmax = pcd[:,2].max() + 0.1
        voxel_origin = [xmin, ymin, zmin]
        x_size = (xmax-xmin) / (N - 1)
        y_size = (ymax-ymin) / (N - 1)
        z_size = (zmax-zmin) / (N - 1)

    overall_index = np.arange(0, N ** 3, 1, dtype=np.int64)
    pts = np.zeros([N ** 3, 3], dtype=np.float32)

    pts[:, 2] = overall_index % N
    pts[:, 1] = (overall_index // N) % N
    pts[:, 0] = ((overall_index // N) // N) % N

    pts[:, 0] = (pts[:, 0] * x_size) + voxel_origin[0]
    pts[:, 1] = (pts[:, 1] * y_size) + voxel_origin[1]
    pts[:, 2] = (pts[:, 2] * z_size) + voxel_origin[2]

    pts = np.vstack((pts.T, np.ones((1, N**3))))

    
    filled = np.ones(pts.shape[1], dtype=bool)

    for calib, transform, im in tqdm(zip(w2cs, projs, silhouette)):
        uvs = transform @ calib @ pts
        uvs[0] = uvs[0] / uvs[2]
        uvs[1] = uvs[1] / uvs[2]
        uvs = np.round(uvs).astype(np.int32)
        x_good = np.logical_and(uvs[0] >= 0, uvs[0] < imgW)
        y_good = np.logical_and(uvs[1] >= 0, uvs[1] < imgH)
        good = np.logical_and(x_good, y_good)
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        res = im[sub_uvs[1, :], sub_uvs[0, :]]
        fill[indices] = res 
        filled = filled & fill.astype(bool)


    occupancy = -filled.astype(np.float32)
    level = -0.5

    occupancy = occupancy.reshape(N,N,N)

    out_name = '%s/%d.obj'%(out_mesh_dire, scan_id)

    convert_sdf_to_mesh(occupancy, voxel_origin, [x_size, y_size, z_size], out_name, level=level)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/nhr_sp.conf')
    parser.add_argument('--scan_id', type=int, default=1)
    args = parser.parse_args()
    main(args.conf, args.scan_id)