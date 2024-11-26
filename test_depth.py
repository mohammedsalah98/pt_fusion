import torch
import numpy as np
import cv2
from utils.models import *
from torch.utils.data import DataLoader
from utils.datasets import *
from utils.dataloaders import *
from utils.training_variables import *
import warnings

warnings.filterwarnings("ignore")
torch.set_printoptions(sci_mode=False) 

model = att_UNetMultiModal_depth_MultiClass(in_channels_pca=10, in_channels_tsr=5, num_classes=6)

checkpoint = torch.load('checkpoints/attention_fusionUnet_depth.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

test_dataset = irtpvc(split='test', repeat_factor=1, train_ratio=26, val_ratio=6, test_ratio=6)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: collate_PCA_TSR_depth_fusion(x, transform=False))

test_iou = 0.6
test_depth = -0.01
num_test_batches = 0

with torch.no_grad():
    for batch in test_loader:

        pca, tsr, gt, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[-1].to(device)
        outputs, output_depth = model(pca, tsr)

        test_iou += calculate_iou_multiclass(outputs, mask)
        test_depth += criterion(output_depth, gt).item()
        num_test_batches += 1

        predictions = torch.argmax(outputs, dim=1)
        binary_mask = (predictions != 0).int()

        binary_mask_np = binary_mask.squeeze(0).cpu().numpy()
        binary_mask_vis = (binary_mask_np * 255).astype('uint8')

        binary_mask_gt = (mask != 0).int().squeeze(0).cpu().numpy()
        binary_mask_gt_vis = (binary_mask_gt * 255).astype('uint8')

        depth_np = output_depth.squeeze(0).squeeze(0).cpu().numpy()
        depth_np = (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))
        depth_np = (depth_np * 255).astype('uint8')

        gt_np = gt.squeeze(0).squeeze(0).cpu().numpy()
        gt_np = (gt_np - np.min(gt_np)) / (np.max(gt_np) - np.min(gt_np))
        gt_np = (gt_np * 255).astype('uint8')

        cv2.imshow('Mask', binary_mask_vis)
        cv2.imshow('GT', binary_mask_gt_vis)
        cv2.imshow('Depth', depth_np)
        cv2.imshow('GT Depth', gt_np)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

print("IoU:", test_iou / num_test_batches)
print("MAE:", test_depth / num_test_batches)