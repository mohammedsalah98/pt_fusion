import torch
import numpy as np
import cv2
from utils.models import *
from torch.utils.data import DataLoader
from utils.datasets import *
from utils.dataloaders import *
import warnings

warnings.filterwarnings("ignore")
torch.set_printoptions(sci_mode=False) 

model = att_UNetMultiModal_MultiClass(in_channels_pca=10, in_channels_tsr=5, num_classes=6)

COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 100, 100),
    2: (100, 255, 100),
    3: (100, 100, 255),
    4: (255, 255, 100),
    5: (255, 100, 255)
}


def apply_color_map(class_mask, color_map):
    h, w = class_mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_image[class_mask == class_id] = color
    return color_image

checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

test_dataset = irtpvc(split='test', repeat_factor=1, train_ratio=26, val_ratio=6, test_ratio=6)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: collate_PCA_TSR_Fusion_MultiClass(x, transform=False))

test_iou = 0.0
test_recall = 0.0
test_precision = 0.0
num_test_batches = 0
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        pca, tsr, gt = batch[0].to(device), batch[1].to(device), batch[-1].to(device)

        outputs = model(pca, tsr)
        predictions = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        ground_truth = gt.squeeze(0).cpu().numpy()

        test_iou += calculate_iou_multiclass(outputs, gt)
        test_recall += calculate_recall(outputs, gt)
        test_precision += calculate_precision(outputs, gt)
        num_test_batches += 1

        prediction_colored = apply_color_map(predictions, COLOR_MAP)
        ground_truth_colored = apply_color_map(ground_truth, COLOR_MAP)

        cv2.imshow('Prediction', prediction_colored)
        cv2.imshow('GT', ground_truth_colored)

        cv2.waitKey(1000)

cv2.destroyAllWindows()

print("IoU:", test_iou / num_test_batches)
print("Recall:", test_recall / num_test_batches)
print("Precision:", test_precision / num_test_batches)