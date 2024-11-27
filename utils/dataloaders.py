import torch
from utils.testing_variables import *
from utils.helper_functions import *

def collate_PCA_TSR_Fusion_MultiClass(batch, transform=True):
    pca_imgs = []
    gt_masks = []
    tsr_imgs = []
    for b in batch:
        pca = b[0].float()
        tsr = b[1].float()[:5]
        gt_mask = b[-1].float() * 2.5
        gt_mask[gt_mask == 2.5] = 5.0
        gt_mask[gt_mask == 2.0] = 4.0
        gt_mask[gt_mask == 1.5] = 3.0
        gt_mask[gt_mask == 1.0] = 2.0
        gt_mask[gt_mask == 0.5] = 1.0
        gt_mask[gt_mask == 0.0] = 0.0
        pca_imgs.append(pca.float())
        tsr_imgs.append(tsr.float())
        gt_masks.append(gt_mask.float())

    pca_imgs = torch.stack(pca_imgs).to(device)
    tsr_imgs = torch.stack(tsr_imgs).to(device)
    gt_masks = torch.stack(gt_masks).unsqueeze(1).to(device)
    
    inputs = torch.cat((pca_imgs, tsr_imgs), dim=1)

    inputs, groundtruth = inputs, gt_masks
    
    pca = inputs[:, :10, :, :]
    tsr = inputs[:, 10:, :, :]

    return pca, tsr, groundtruth.squeeze(1).long()

def collate_PCA_TSR_depth_fusion(batch, transform=True):
    pca_imgs = []
    tsr_imgs = []
    depth_maps = []
    for b in batch:
        pca = b[0].float()
        tsr = b[1].float()[:5]
        depth_map = b[-1].float() * 2.5
        pca_imgs.append(pca.float())
        tsr_imgs.append(tsr.float())
        depth_maps.append(depth_map.float())

    pca_imgs = torch.stack(pca_imgs).to(device)
    tsr_imgs = torch.stack(tsr_imgs).to(device)
    depth_maps = torch.stack(depth_maps).unsqueeze(1).to(device)

    inputs = torch.cat((pca_imgs, tsr_imgs), dim=1)

    inputs, groundtruth = inputs, depth_maps

    gt_masks = []
    for gt_mask in groundtruth:
        gt_mask[gt_mask == 2.5] = 5.0
        gt_mask[gt_mask == 2.0] = 4.0
        gt_mask[gt_mask == 1.5] = 3.0
        gt_mask[gt_mask == 1.0] = 2.0
        gt_mask[gt_mask == 0.5] = 1.0
        gt_mask[gt_mask == 0.0] = 0.0
        gt_masks.append(gt_mask)
    gt_masks = torch.stack(gt_masks).to(device)

    pca = inputs[:, :10, :, :]
    tsr = inputs[:, 10:, :, :]
    
    return pca, tsr, groundtruth, gt_masks.squeeze(1).long()