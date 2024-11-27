import torch
from torch.utils.data import DataLoader
from utils.datasets import *
from utils.dataloaders import *
import warnings
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from utils.helper_functions import *

warnings.filterwarnings("ignore")

model = att_UNetMultiModal_MultiClass(in_channels_pca=10, in_channels_tsr=5, num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
weighing_factor = 0.5
num_epochs = 30
batch_size = 8

train_dataset = irtpvc_augmented(split='train', augment_factor=500, train_ratio=26, val_ratio=6, test_ratio=6)
val_dataset = irtpvc_augmented(split='val', augment_factor=500, train_ratio=26, val_ratio=6, test_ratio=6)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_PCA_TSR_Fusion_MultiClass(x, transform=True))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_PCA_TSR_Fusion_MultiClass(x, transform=True))

model = model.to(device)

train_losses = []
val_losses = []
test_losses = []

train_ious = []
val_ious = []
test_ious = []

train_rec = []
val_rec = []
test_rec = []

train_pre = []
val_pre = []
test_pre = []

plt.ion()
fig_loss, ax_loss = plt.subplots()
line1_loss, = ax_loss.plot([], [], label="Training Loss")
line2_loss, = ax_loss.plot([], [], label="Validation Loss")
ax_loss.set_xlim(0, num_epochs)
ax_loss.set_ylim(0, 2)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

fig_iou, ax_iou = plt.subplots()
line1_iou, = ax_iou.plot([], [], label="Training IoU")
line2_iou, = ax_iou.plot([], [], label="Validation IoU")
ax_iou.set_xlim(0, num_epochs)
ax_iou.set_ylim(0, 1)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("IoU")

fig_rec, ax_rec = plt.subplots()
line1_rec, = ax_rec.plot([], [], label="Training Recall")
line2_rec, = ax_rec.plot([], [], label="Validation Recall")
ax_rec.set_xlim(0, num_epochs)
ax_rec.set_ylim(0, 1)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Recall")

fig_pre, ax_pre = plt.subplots()
line1_pre, = ax_pre.plot([], [], label="Training Precision")
line2_pre, = ax_pre.plot([], [], label="Validation Precision")
ax_pre.set_xlim(0, num_epochs)
ax_pre.set_ylim(0, 1)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Precision")

for epoch in trange(num_epochs, desc='training'): 
    # Train
    model.train()
    training_loss = 0.0
    training_iou = 0.0
    training_recall = 0.0
    training_precision = 0.0
    num_batches = 0
    with tqdm(train_loader, unit='batch', desc=f'Epoch {epoch+1} [Training]') as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            pca, tsr, gt = batch[0].to(device), batch[1].to(device), batch[-1].to(device)
            outputs = model(pca, tsr)

            loss = criterion(outputs, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_iou += calculate_iou_multiclass(outputs, gt)
            training_recall += calculate_recall(outputs, gt)
            training_precision += calculate_precision(outputs, gt)

            num_batches += 1

            pfix = {"train_loss": training_loss / num_batches, "train_IoU": f"{(training_iou / num_batches):.5f}", \
                    "train_recall": f"{(training_recall / num_batches):.2f}", \
                    "train_precision": f"{(training_precision / num_batches):.2f}"}
            tepoch.set_postfix(pfix)
    
    train_losses.append(training_loss / num_batches)
    train_rec.append(training_recall / num_batches)
    train_ious.append(training_iou / num_batches)
    train_pre.append(training_precision / num_batches)
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_recall = 0.0
    val_precision = 0.0
    num_val_batches = 0
    with torch.no_grad():
        with tqdm(val_loader, unit='batch', desc=f'Epoch {epoch+1} [Validation]') as vepoch:
            for batch_idx, batch in enumerate(vepoch):
                pca, tsr, gt = batch[0].to(device), batch[1].to(device), batch[-1].to(device)
                outputs = model(pca, tsr)
                loss = criterion(outputs, gt)

                val_loss += loss.item()
                val_iou += calculate_iou_multiclass(outputs, gt)
                val_recall += calculate_recall(outputs, gt)
                val_precision += calculate_precision(outputs, gt)
                num_val_batches += 1

                pfix = {"val_loss": val_loss / num_val_batches, "val_IoU": f"{(val_iou / num_val_batches):.5f}", \
                        "val_recall": f"{(val_recall / num_val_batches):.2f}", \
                        "val_precision": f"{(val_precision / num_val_batches):.2f}"}
                vepoch.set_postfix(pfix)
    
    val_losses.append(val_loss / num_val_batches)
    val_ious.append(val_iou / num_val_batches)
    val_rec.append(val_recall / num_val_batches)
    val_pre.append(val_precision / num_val_batches)

    line1_loss.set_xdata(range(1, len(train_losses) + 1))
    line1_loss.set_ydata(train_losses)
    
    line2_loss.set_xdata(range(1, len(val_losses) + 1))
    line2_loss.set_ydata(val_losses)
    
    ax_loss.set_ylim(0, max(max(train_losses), max(val_losses), max(test_losses)) + 0.1)
    fig_loss.canvas.draw()
    fig_loss.canvas.flush_events()

    line1_iou.set_xdata(range(1, len(train_ious) + 1))
    line1_iou.set_ydata(train_ious)
    
    line2_iou.set_xdata(range(1, len(val_ious) + 1))
    line2_iou.set_ydata(val_ious)
    
    ax_iou.set_ylim(0, max(max(train_ious), max(val_ious), max(test_ious)) + 0.1)
    fig_iou.canvas.draw()
    fig_iou.canvas.flush_events()

    line1_rec.set_xdata(range(1, len(train_rec) + 1))
    line1_rec.set_ydata(train_rec)
    
    line2_rec.set_xdata(range(1, len(val_rec) + 1))
    line2_rec.set_ydata(val_rec)
    
    ax_rec.set_ylim(0, max(max(train_rec), max(val_rec), max(test_rec)) + 0.1)
    fig_rec.canvas.draw()
    fig_rec.canvas.flush_events()

    line1_pre.set_xdata(range(1, len(train_pre) + 1))
    line1_pre.set_ydata(train_pre)
    
    line2_pre.set_xdata(range(1, len(val_pre) + 1))
    line2_pre.set_ydata(val_pre)
    
    ax_pre.set_ylim(0, max(max(train_pre), max(val_pre), max(test_pre)) + 0.1)
    fig_rec.canvas.draw()
    fig_rec.canvas.flush_events()

plt.ioff()
plt.show()