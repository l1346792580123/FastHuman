import os
from os.path import join
from glob import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import load_K_Rt_from_P


def campose_to_extrinsic(camposes):
    if camposes.shape[1]!=12:
        raise Exception(" wrong campose data structure!")
        return
    
    res = np.zeros((camposes.shape[0],4,4))
    
    res[:,0:3,2] = camposes[:,0:3]
    res[:,0:3,0] = camposes[:,3:6]
    res[:,0:3,1] = camposes[:,6:9]
    res[:,0:3,3] = camposes[:,9:12]
    res[:,3,3] = 1.0
    
    return res

def read_intrinsics(fn_instrinsic):
    fo = open(fn_instrinsic)
    data= fo.readlines()
    i = 0
    Ks = []
    while i<len(data):
        if len(data[i])>5:
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            a = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            b = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            c = np.array(tmp)
            res = np.vstack([a,b,c])
            Ks.append(res)

        i = i+1
    Ks = np.stack(Ks)
    fo.close()

    return Ks


def get_nhr_data(data_path, scan_id, num, res=(1224,1024), shape=(1024,1224)):
    c2ws = campose_to_extrinsic(np.loadtxt(join(data_path,'CamPose.inf')))
    all_w2cs = np.linalg.inv(c2ws)
    all_projs = read_intrinsics(join(data_path, 'Intrinsic.inf'))
    h, w = shape
    imgs = []
    grayimgs = []
    masks = []
    w2cs = []
    projs = []
    for i in range(num):
        img = cv2.imread('%s/img/%d/img_%04d.jpg'%(data_path,scan_id,i))
        if img.shape[:2] == shape:
            mask = cv2.imread('%s/img/%d/mask/img_%04d.jpg'%(data_path,scan_id,i))[:,:,0]
            mask = (mask>127.5).astype(np.float32)
            img[mask==0] = 0
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, res)
            grayimg = cv2.resize(grayimg, res)
            mask = cv2.resize(mask, res, interpolation=cv2.INTER_NEAREST)

            img = torch.from_numpy((img/255.)).float().cuda()
            grayimg = torch.from_numpy((grayimg/255.)).float().cuda()
            mask  = torch.from_numpy(mask).float().cuda()

            imgs.append(img)
            grayimgs.append(grayimg)
            masks.append(mask)

            w2cs.append(all_w2cs[i].astype(np.float32))

            proj = np.zeros([4,4])
            proj[0,0] = all_projs[i,0,0] / (w/2)
            proj[0,1] = all_projs[i,0,1] / (w/2)
            proj[0,2] = all_projs[i,0,2] / (w/2) - 1.
            proj[1,1] = all_projs[i,1,1] / (h/2)
            proj[1,2] = all_projs[i,1,2] / (h/2) - 1.
            proj[2,2] = 0.
            proj[2,3] = -1.
            proj[3,2] = 1.0
            proj[3,3] = 0.0
            projs.append(proj.astype(np.float32))

    w2cs = torch.from_numpy(np.stack(w2cs)).permute(0,2,1).cuda()
    projs = torch.from_numpy(np.stack(projs)).permute(0,2,1).cuda()
    imgs = torch.stack(imgs, dim=0)
    grayimgs = torch.stack(grayimgs, dim=0)
    masks = torch.stack(masks, dim=0)

    return imgs, grayimgs, masks, w2cs, projs



def get_dtu_data(data_path, scan_id, res=(1600,1200)):
    camera_dict = np.load(join(data_path, 'scan%d/imfunc4/cameras_hd.npz'%scan_id))
    num = 49

    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(num)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(num)]


    projs = []
    w2cs = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        proj, w2c = load_K_Rt_from_P(P)

        proj[0,0] = proj[0,0] / 800.
        proj[0,1] = proj[0,1] / 800.
        proj[0,2] = proj[0,2] / 800. - 1.
        proj[1,1] = proj[1,1] / 600.
        proj[1,2] = proj[1,2] / 600. - 1.
        proj[2,2] = 0.
        proj[2,3] = -1.
        proj[3,2] = 1.0
        proj[3,3] = 0.0

        proj = torch.from_numpy(proj.astype(np.float32)).cuda()
        w2c = torch.from_numpy(w2c.astype(np.float32)).cuda()

        projs.append(proj)
        w2cs.append(w2c)

    # transpose for right multiplication
    w2cs = torch.stack(w2cs, dim=0).permute(0,2,1).contiguous()
    projs = torch.stack(projs, dim=0).permute(0,2,1).contiguous()

    imgs = []
    grayimgs = []
    masks = []
    for i in range(num):
        img = cv2.imread(join(data_path, 'scan%d/imfunc4/image_hd/%06d.png'%(scan_id, i)))
        mask = cv2.imread(join(data_path, 'scan%d/imfunc4/pmask/%03d.png'%(scan_id,i)))[:,:,0]
        mask = (mask>0).astype(np.float32)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # grayimg = cv2.equalizeHist(grayimg)
        img = cv2.resize(img, res)
        grayimg = cv2.resize(grayimg, res)
        mask = cv2.resize(mask, res, interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy((img/255.)).float().cuda()
        grayimg = torch.from_numpy((grayimg/255.)).float().cuda()
        mask  = torch.from_numpy(mask).float().cuda()

        imgs.append(img)
        grayimgs.append(grayimg)
        masks.append(mask)

    imgs = torch.stack(imgs, dim=0)
    grayimgs = torch.stack(grayimgs, dim=0)
    masks = torch.stack(masks, dim=0)


    return imgs, grayimgs, masks, w2cs, projs
