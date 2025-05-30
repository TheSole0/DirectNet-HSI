import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from model import DirectNet
from utils import init_weights
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


def visualize_anomaly_with_bbox(x, anomaly_map, threshold=None, percentile=99.5, bbox_min_area=10):
    rgb_idx = [111, 82, 26] if x.shape[2] > 112 else [x.shape[2]//2, x.shape[2]//4, x.shape[2]//8]
    rgb = (x[:, :, rgb_idx] - x[:, :, rgb_idx].min()) / (x[:, :, rgb_idx].ptp() + 1e-8)

    if threshold is None:
        threshold = np.percentile(anomaly_map, percentile)
    anomaly_mask = anomaly_map > threshold

    labeled_mask = label(anomaly_mask)
    props = regionprops(labeled_mask)

    disp_rgb = rgb.copy()
    disp_rgb[~anomaly_mask] = 0.15 * disp_rgb[~anomaly_mask]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(disp_rgb)
    ax.imshow(np.ma.masked_where(~anomaly_mask, anomaly_mask), cmap='autumn', alpha=0.45)
    for prop in props:
        if prop.area < bbox_min_area:
            continue
        minr, minc, maxr, maxc = prop.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='lime', linewidth=2)
        ax.add_patch(rect)
    ax.set_title(f"Anomaly Detection (Threshold={threshold:.4f})")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    return anomaly_mask, labeled_mask


def run_predict_direct(
    dataset_list,
    config,
    device="cuda:0",
    ckpt_path=None,
    visualize=False,
    percentile=99.5
):
    device = torch.device(device)
    bands = dataset_list[0]["corrected"].shape[2]

    # 모델 설정
    win_out = config.get("win_out", 19)
    net = DirectNet(
        bands, bands,
        nch_ker=config.get("nch_ker", 64),
        norm=config.get("norm_mode", "bnorm"),
        nblk=(win_out - 7) // 4
    ).to(device)

    # 체크포인트 로딩
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {ckpt_path}")
    print(f"✅ Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    net.load_state_dict(state_dict)
    net.eval()

    results = []
    for idx, d in enumerate(dataset_list):
        x = d["corrected"].astype(np.float32)
        y = d["new_label"].astype(np.int64)

        # [1] new_label 분포
        unique, counts = np.unique(y, return_counts=True)
        print(f"[{idx}] new_label 분포: {dict(zip(unique, counts))}")

        # [2] mask, gt 분포
        mask = (y != -1)
        gt = (y == 1).astype(np.uint8)
        print(f"[{idx}] mask 픽셀 수 (유효): {np.sum(mask)} / 전체: {mask.size}")
        print(f"[{idx}] 이상(1) 픽셀 수: {np.sum(gt)}")
        print(f"[{idx}] 무시(-1) 픽셀 수: {np.sum(y == -1)}")

        # [3] 네트워크 예측
        x_tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out_tensor = net(x_tensor)
        recon = out_tensor.squeeze(0).cpu().numpy()
        recon = np.transpose(recon, (1, 2, 0))
        diff = np.abs(recon - x)
        anomaly_map = np.mean(diff, axis=2)

        # [4] threshold 및 AUC
        valid_anomaly_map = anomaly_map[mask]
        threshold = np.percentile(valid_anomaly_map, percentile)
        anomaly_mask = anomaly_map > threshold
        gt_flat = gt[mask].flatten()
        pred_flat = anomaly_map[mask].flatten()
        auc = roc_auc_score(gt_flat, pred_flat) if np.unique(gt_flat).size > 1 else np.nan

        # [5] 시각화
        if visualize:
            visualize_anomaly_with_bbox(x, anomaly_map, threshold=threshold, bbox_min_area=10)

        results.append({
            "path": d.get("path", ""),
            "anomaly_map": anomaly_map,
            "auc": auc,
            "recon": recon,
            "gt": gt,
        })
        print(f"[{idx}] {d.get('path','')} - AUC: {auc:.4f}")

    return results
