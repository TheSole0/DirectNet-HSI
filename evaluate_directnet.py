import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def evaluate_dataset_direct(
    dataset_list,
    config,
    model_path,
    device="cuda:0",
    visualize=False,
    out_dir=None,
    percentile=98,
    save=False
):
    device = torch.device(device)
    bands = dataset_list[0]["corrected"].shape[2]

    # 모델 준비
    from model import DirectNet
    net = DirectNet(
        bands, bands,
        nch_ker=config.get("nch_ker", 64),
        norm=config.get("norm_mode", "bnorm"),
        nblk=(config.get("win_out", 19) - 7) // 4
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    all_auc = []
    all_cm = []
    sample_reports = []

    for idx, d in enumerate(dataset_list):
        x = d["corrected"].astype(np.float32)
        y = d["new_label"].astype(np.int64)
        mask = (y != -1)
        gt = (y == 1).astype(np.uint8)

        # 네트워크 예측
        x_tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out_tensor = net(x_tensor)
        recon = out_tensor.squeeze(0).cpu().numpy()
        recon = np.transpose(recon, (1, 2, 0))
        diff = np.abs(recon - x)
        anomaly_map = np.mean(diff, axis=2)

        # threshold 산정 (마스킹된 영역만)
        valid_anomaly_map = anomaly_map[mask]
        threshold = np.percentile(valid_anomaly_map, percentile)

        # anomaly mask 예측 (이상 픽셀 = 1)
        pred_mask = anomaly_map > threshold
        pred_label = (pred_mask & mask)
        pred_flat = anomaly_map[mask].flatten()
        gt_flat = gt[mask].flatten()

        auc = roc_auc_score(gt_flat, pred_flat) if np.unique(gt_flat).size > 1 else np.nan
        all_auc.append(auc)
        cm = confusion_matrix(gt_flat, pred_mask[mask].astype(np.uint8), labels=[0, 1])
        all_cm.append(cm)

        sample_reports.append({
            "path": d.get("path", ""),
            "auc": auc,
            "threshold": threshold,
            "cm": cm,
            "n_total": int(mask.sum()),
            "n_anomaly": int((gt == 1).sum()),
            "n_pred_anomaly": int(pred_mask[mask].sum())
        })

        if visualize:
            print(f"[{idx}] {d.get('path','')} - AUC: {auc:.4f}, threshold={threshold:.5f}")

        if save and out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            np.savez(os.path.join(out_dir, f"anomaly_map_{idx}.npz"),
                     anomaly_map=anomaly_map, recon=recon, label=y)

    oa = np.nanmean(all_auc)
    rpt = {
        "mean_auc": oa,
        "sample_reports": sample_reports,
        "confusion_matrices": all_cm
    }
    return oa, rpt
