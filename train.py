import os
import json
import csv
import time
import random
import warnings
import subprocess
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import kornia
from config import Config

opt = Config('training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_RGB_mfnet import get_training_data
from MMFNet import MMFNet
from warmup_scheduler import GradualWarmupScheduler
from utils.seg_util import *
from utils.dice import *
from utils.MEF_SSIM_loss import th_SSIM_LOSS, Y_Upper

warnings.filterwarnings("ignore", category=UserWarning)

def rgb_to_ycbcr(img):
    return kornia.color.rgb_to_ycbcr(img)

def init_history():
    return {
        'epoch': [], 'lr': [], 'loss_total': [], 'loss_ir': [], 'loss_rgb': [],
        'loss_rec_ir': [], 'loss_rec_rgb': [],
        'loss_rec_ir_grad': [], 'loss_rec_ir_lap': [],
        'loss_rec_rgb_grad': [], 'loss_rec_rgb_lap': [],
        'loss_seg': []
    }

def load_history_if_exists(log_dir):
    path = os.path.join(log_dir, 'loss_history.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return init_history()

def save_history(history, log_dir):
    json_path = os.path.join(log_dir, 'loss_history.json')
    csv_path = os.path.join(log_dir, 'loss_history.csv')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    keys = list(history.keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in zip(*[history[k] for k in keys]):
            writer.writerow(row)

def plot_history(history, log_dir):
    epochs = history.get('epoch', [])
    if not epochs:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss_total'], label='loss_total')
    plt.plot(epochs, history['loss_ir'], label='loss_ir')
    plt.plot(epochs, history['loss_rgb'], label='loss_rgb')
    plt.legend(); plt.grid(True); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.tight_layout(); plt.savefig(os.path.join(log_dir, 'loss_total_curve.png'), dpi=300); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss_rec_ir_grad'], label='loss_rec_ir_grad')
    plt.plot(epochs, history['loss_rec_ir_lap'], label='loss_rec_ir_lap')
    plt.plot(epochs, history['loss_rec_rgb_grad'], label='loss_rec_rgb_grad')
    plt.plot(epochs, history['loss_rec_rgb_lap'], label='loss_rec_rgb_lap')
    plt.legend(); plt.grid(True); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.tight_layout(); plt.savefig(os.path.join(log_dir, 'loss_reconstruction_edge_terms_curve.png'), dpi=300); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['lr'], label='learning_rate')
    plt.legend(); plt.grid(True); plt.xlabel('Epoch'); plt.ylabel('LR')
    plt.tight_layout(); plt.savefig(os.path.join(log_dir, 'lr_curve.png'), dpi=300); plt.close()

def append_text_log(log_dir, text):
    with open(os.path.join(log_dir, 'train_console_log.txt'), 'a', encoding='utf-8') as f:
        f.write(text + '\n')

def gradient_map(x):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    c = x.shape[1]
    gx = F.conv2d(x, sobel_x.repeat(c,1,1,1), padding=1, groups=c)
    gy = F.conv2d(x, sobel_y.repeat(c,1,1,1), padding=1, groups=c)
    return torch.abs(gx) + torch.abs(gy)

def laplacian_map(x):
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    c = x.shape[1]
    return F.conv2d(x, lap.repeat(c,1,1,1), padding=1, groups=c)

def gradient_loss(pred, target):
    return F.l1_loss(gradient_map(pred), gradient_map(target))

def laplacian_loss(pred, target):
    return F.l1_loss(laplacian_map(pred), laplacian_map(target))

if __name__ == '__main__':
    random.seed(1234); np.random.seed(1234)
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)

    data = opt.Datasets.data
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, data, 'models'); utils.mkdir(model_dir)
    log_dir = os.path.join(opt.TRAINING.SAVE_DIR, data, 'logs'); utils.mkdir(log_dir)
    train_dir = opt.TRAINING.TRAIN_DIR

    model = MMFNet(opt.TRAINING.NUM_CLASSES).cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    start_epoch = 0
    if opt.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, 'model_latest.pth')
        utils.load_checkpoint(model, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest)
        utils.load_optim(optimizer, path_chk_rest)
        for _ in range(1, start_epoch):
            scheduler.step()

    l1_criterion = nn.L1Loss()
    CE_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    Dice_criterion = DiceLoss(smooth=0.05, ignore_index=255)
    train_dataset = get_training_data(train_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True, pin_memory=True)
    history = load_history_if_exists(log_dir) if opt.TRAINING.RESUME else init_history()

    lambda_grad_rec = 0.4
    lambda_lap_rec = 0.1
    lambda_lap_sr = lambda_lap_rec  # Laplacian weight for self-reconstruction.

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        model.train()

        sums = {k: 0.0 for k in ['loss','ir','rgb','rec_ir','rec_rgb','rec_ir_grad','rec_ir_lap','rec_rgb_grad','rec_rgb_lap','seg']}

        for data_batch in tqdm(train_loader, 0):
            optimizer.zero_grad(set_to_none=True)
            target_ir = data_batch[0].cuda()
            input_ir = data_batch[1].cuda()
            target_rgb = data_batch[2].cuda()
            input_rgb = data_batch[3].cuda()
            target_seg = data_batch[4].cuda()

            input_ycbcr = rgb_to_ycbcr(input_rgb)
            target = Y_Upper(target_ir[:, :1, :, :], input_ycbcr[:, :1, :, :], 1.7).cuda()

            res_ir, res_y, res_rec_ir, res_rec_y, res_seg, _ = model(input_ir[:, :1, :, :], input_ycbcr[:, :1, :, :])

            res_ycbcr = torch.cat((torch.clamp(res_y, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]), dim=1)
            res_rgb = kornia.color.ycbcr_to_rgb(res_ycbcr)

            res_rec_ycbcr = torch.cat((torch.clamp(res_rec_y, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]), dim=1)
            res_rec_rgb = kornia.color.ycbcr_to_rgb(res_rec_ycbcr)

            # Match the modality targets used by the cross-reconstruction Laplacian terms.
            loss_ir_grad = gradient_loss(res_ir, target_ir[:, :1, :, :])
            loss_rgb_grad = gradient_loss(res_y, input_ycbcr[:, :1, :, :])
            loss_ir_lap = laplacian_loss(res_ir, target_ir[:, :1, :, :])
            loss_rgb_lap = laplacian_loss(res_y, input_ycbcr[:, :1, :, :])
            loss_ir = l1_criterion(res_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_ir) + lambda_grad_rec * loss_ir_grad + lambda_lap_sr * loss_ir_lap
            loss_rgb = l1_criterion(res_rgb, target_rgb) + th_SSIM_LOSS(target, res_y) + lambda_grad_rec * loss_rgb_grad + lambda_lap_sr * loss_rgb_lap

            loss_rec_ir_grad = gradient_loss(res_rec_ir, target_ir[:, :1, :, :])
            loss_rec_ir_lap = laplacian_loss(res_rec_ir, target_ir[:, :1, :, :])
            loss_rec_rgb_grad = gradient_loss(res_rec_y, input_ycbcr[:, :1, :, :])
            loss_rec_rgb_lap = laplacian_loss(res_rec_y, input_ycbcr[:, :1, :, :])

            loss_rec_ir = l1_criterion(res_rec_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_rec_ir) + lambda_grad_rec * loss_rec_ir_grad + lambda_lap_rec * loss_rec_ir_lap
            loss_rec_rgb = l1_criterion(res_rec_rgb, target_rgb) + th_SSIM_LOSS(target, res_rec_y) + lambda_grad_rec * loss_rec_rgb_grad + lambda_lap_rec * loss_rec_rgb_lap
            loss_seg = Dice_criterion(res_seg, target_seg) + CE_criterion(res_seg, target_seg)

            loss = loss_ir + loss_rgb + loss_rec_ir + loss_rec_rgb + loss_seg
            loss.backward()
            optimizer.step()

            sums['loss'] += loss.item(); sums['ir'] += loss_ir.item(); sums['rgb'] += loss_rgb.item()
            sums['rec_ir'] += loss_rec_ir.item(); sums['rec_rgb'] += loss_rec_rgb.item()
            sums['rec_ir_grad'] += loss_rec_ir_grad.item(); sums['rec_ir_lap'] += loss_rec_ir_lap.item()
            sums['rec_rgb_grad'] += loss_rec_rgb_grad.item(); sums['rec_rgb_lap'] += loss_rec_rgb_lap.item()
            sums['seg'] += loss_seg.item()

        n = max(len(train_loader), 1)
        history['epoch'].append(int(epoch)); history['lr'].append(float(current_lr))
        history['loss_total'].append(float(sums['loss']/n)); history['loss_ir'].append(float(sums['ir']/n)); history['loss_rgb'].append(float(sums['rgb']/n))
        history['loss_rec_ir'].append(float(sums['rec_ir']/n)); history['loss_rec_rgb'].append(float(sums['rec_rgb']/n))
        history['loss_rec_ir_grad'].append(float(sums['rec_ir_grad']/n)); history['loss_rec_ir_lap'].append(float(sums['rec_ir_lap']/n))
        history['loss_rec_rgb_grad'].append(float(sums['rec_rgb_grad']/n)); history['loss_rec_rgb_lap'].append(float(sums['rec_rgb_lap']/n))
        history['loss_seg'].append(float(sums['seg']/n))

        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'model_epoch_{epoch}.pth'))
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, 'model_latest.pth'))

        save_history(history, log_dir); plot_history(history, log_dir)

        msg = f"Epoch {epoch} | time {time.time()-epoch_start_time:.2f}s | total {sums['loss']/n:.6f} | ir {sums['ir']/n:.6f} | rgb {sums['rgb']/n:.6f} | rec_ir {sums['rec_ir']/n:.6f} | rec_rgb {sums['rec_rgb']/n:.6f} | rec_ir_grad {sums['rec_ir_grad']/n:.6f} | rec_ir_lap {sums['rec_ir_lap']/n:.6f} | rec_rgb_grad {sums['rec_rgb_grad']/n:.6f} | rec_rgb_lap {sums['rec_rgb_lap']/n:.6f} | seg {sums['seg']/n:.6f} | lr {current_lr:.8f}"
        print(msg); append_text_log(log_dir, msg)
        scheduler.step()

    append_text_log(log_dir, 'Training completed. Starting automatic test and metric evaluation...')
    try:
        test_dataset = os.path.basename(os.path.normpath(train_dir))
        latest_weight = os.path.join(model_dir, 'model_latest.pth')
        result_root = os.path.join(opt.TRAINING.SAVE_DIR, data, 'results_auto')
        metric_log_dir = os.path.join(opt.TRAINING.SAVE_DIR, data, 'logs')

        subprocess.run([sys.executable, 'test_mfnet.py', '--dataset', test_dataset, '--input_dir', './dataset/test', '--result_dir', result_root, '--weights', latest_weight, '--gpus', gpus, '--num_classes', str(opt.TRAINING.NUM_CLASSES)], check=True)
        subprocess.run([sys.executable, './test_metric/test_metric.py', '--dataset', test_dataset, '--fus_dir', os.path.join(result_root, test_dataset), '--save_dir', metric_log_dir], check=True)
        append_text_log(log_dir, 'Auto test and metric evaluation finished successfully.')
    except Exception as e:
        err = f'Auto test/metric failed: {e}'
        print(err); append_text_log(log_dir, err)
