import os
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time

from model import DirectNet
from utils import setup_seed, init_weights

# 1. 기존 훈련 Trainer 유지 (dataloader만 교체)
class Trainer(object):
    def __init__(self, opt, model, criterion, optimizer, dataloader, device, model_path, logs_path, save_freq=50, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.model_path = model_path
        self.logs_path = logs_path
        self.save_freq = save_freq
        self.scheduler = scheduler
        self.opt = opt
        self.epochs = self.opt.epochs if hasattr(self.opt, "epochs") else self.opt.get("epochs", 100)
        self.dataset = self.opt.dataset if hasattr(self.opt, "dataset") else self.opt.get("dataset", "MyData")

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        self.log_output = open(f"{self.logs_path}/log.txt", 'w')
        self.writer = SummaryWriter(self.logs_path)
        print(self.opt)
        print(self.opt, file=self.log_output)

    def train_epoch(self):
        self.model.train(True)
        loss_train = []
        for i, data in enumerate(self.dataloader):
            label = data['label'].to(self.device).float()
            input = data['input'].to(self.device).float()
            mask  = data['mask'].to(self.device)
            output = self.model(input)
            loss = self.criterion(output[~mask], label[~mask])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_train.append(loss.item())
            # 미니배치 단위 loss 로그 추가


            
        # 에폭 단위 평균 loss
        print(f"[Epoch {self.epoch+1}] loss = {np.mean(loss_train):.6f}")
        info = { 'Loss_train': np.mean(loss_train) }
        for tag, value in info.items():
            self.writer.add_scalar(tag, value, self.epoch + 1)
        if ((self.epoch + 1) % self.save_freq == 0 or (self.epoch + 1) == self.epochs):
            torch.save(self.model.state_dict(),
                       os.path.join(self.model_path, f'DirectNet_{self.dataset}_{self.epoch+1}.pkl'))


    def train(self, epoches=None):
        if epoches is None:
            epoches = self.opt.epochs
        self.epochs = epoches  # 반드시 실제 반복 epoch 수로 덮어쓰기
        for epoch in range(epoches):
            self.epoch = epoch
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, epoches))
            print('Epoch {}/{}'.format(epoch + 1, epoches), file = self.log_output)
            print('-' * 50)
            self.train_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
        self.log_output.close()
        return self.model


# 2. Dataset (dataset_list 지원)
class DirectNetDatasetList(torch.utils.data.Dataset):
    def __init__(self, dataset_list, augment=False, augment_params=None):
        self.dataset_list = dataset_list
        self.augment = augment
        self.augment_params = augment_params or {}

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        d = self.dataset_list[idx]

        x = d["corrected"].astype(np.float32)        # (H, W, C)
        y = d["new_label"].astype(np.int64)          # (-1,0,1)

        if self.augment:
            x = self.apply_augment(x)

        # ── 1) 입력 / 타깃 ───────────────────────────
        inp   = np.transpose(x, (2, 0, 1)).copy()    # (C,H,W)
        target = inp.copy()                          # 복원 대상 = 원본

        # ── 2) 손실 계산 마스크 ──────────────────────
        #   정상(0) 픽셀만 손실 계산, 이상(1)·무시(-1)는 제외
        ignore_mask = np.broadcast_to((y == -1), inp.shape).copy()


        return {
            "input": torch.from_numpy(inp).float(),
            "label": torch.from_numpy(target).float(),
            "mask":  torch.from_numpy(ignore_mask)   # bool
        }


    def apply_augment(self, x):
        # Gaussian noise
        if self.augment_params.get("noise_prob", 0) > np.random.rand():
            std = self.augment_params.get("noise_std", 0.01)
            x = x + np.random.normal(0, std, x.shape)
        # Band shift
        if self.augment_params.get("shift_prob", 0) > np.random.rand():
            max_shift = self.augment_params.get("max_shift", 3)
            shift = np.random.randint(-max_shift, max_shift + 1)
            x = np.roll(x, shift, axis=2)
        # Band dropout
        if self.augment_params.get("dropout_prob", 0) > np.random.rand():
            band_dropout_prob = self.augment_params.get("band_dropout_prob", 0.05)
            mask = np.random.rand(x.shape[2]) > band_dropout_prob
            x = x * mask
        return x.astype(np.float32)

# 3. 외부에서 불러서 사용할 함수 (입력: config, dataset_list 등)
def train_directnet_with_datasetlist(
    config,
    dataset_list,
    epoches=100,
    device="cuda:0",
    augment=False,
    augment_params=None
):
    seed = getattr(config, "seed", 42) if hasattr(config, "seed") else config.get("seed", 42)
    setup_seed(seed)
    device = torch.device(device)

    # dataloader
    dataset = DirectNetDatasetList(dataset_list, augment=augment, augment_params=augment_params)
    batch_size = getattr(config, "batch_size", 1) if hasattr(config, "batch_size") else config.get("batch_size", 1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    bands = dataset_list[0]["corrected"].shape[2]
    win_out = getattr(config, "win_out", 19) if hasattr(config, "win_out") else config.get("win_out", 19)
    net = DirectNet(bands, bands, nch_ker=getattr(config, "nch_ker", 64), norm=getattr(config, "norm_mode", "bnorm"),
                    nblk=(win_out - 7) // 4).to(device)

    init_type = getattr(config, "init_weight_type", "normal") if hasattr(config, "init_weight_type") else config.get("init_weight_type", "normal")
    init_gain = getattr(config, "init_gain", 0.02) if hasattr(config, "init_gain") else config.get("init_gain", 0.02)
    init_weights(net, init_type=init_type, init_gain=init_gain)

    optimizer = optim.Adam(net.parameters(), lr=getattr(config, "learning_rate", 1e-4), betas=(0.5, 0.999),
                           weight_decay=getattr(config, "weight_decay", 1e-5))
    scheduler = None

    loss_mode = getattr(config, "loss_mode", "l1") if hasattr(config, "loss_mode") else config.get("loss_mode", "l1")
    if loss_mode == "l1":
        criterion = nn.L1Loss()
    elif loss_mode == "l2":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")

    DB = getattr(config, "dataset", "MyData") if hasattr(config, "dataset") else config.get("dataset", "MyData")
    expr_dir = os.path.join('./checkpoints/', DB)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    prefix = 'DirectNet' + '_batch_size_' + str(batch_size) + '_epoch_' + str(epoches) + \
        '_learning_rate_' + str(getattr(config, "learning_rate", 1e-4)) + \
        '_win_in_' + str(getattr(config, "win_in", 3)) + '_win_out_' + str(win_out) + \
        '_gpu_ids_' + str(getattr(config, "gpu_ids", 0))
    trainfile = os.path.join(expr_dir, prefix)
    if not os.path.exists(trainfile):
        os.makedirs(trainfile)
    model_path = os.path.join(trainfile, 'model')
    logs_path = os.path.join(trainfile, './logs')

    save_freq = getattr(config, "save_freq", 10) if hasattr(config, "save_freq") else config.get("save_freq", 10)

    print(f">>> Training samples: {len(dataset_list)} | Model save dir: {model_path}")

    trainer = Trainer(config, net, criterion, optimizer, loader, device, model_path, logs_path, save_freq=save_freq)
    trainer.train(epoches)
    print("Training finished.")
    
    torch.cuda.empty_cache()   # PyTorch GPU 캐시 해제

