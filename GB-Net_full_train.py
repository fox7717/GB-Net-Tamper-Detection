import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import resnet50
import os
import requests
import zipfile
from PIL import Image
from tqdm import tqdm

# ===================== 1. 新增：CASIA数据集自动下载+格式转换（GitHub直接用） =====================
def download_casia_github():
    # CASIA数据集开源地址（Kaggle镜像，GitHub可访问）
    download_url = "https://github.com/namtpham/casia2groundtruth/archive/refs/heads/master.zip"
    save_path = "./casia_dataset.zip"
    extract_path = "./casia_raw"
    final_dataset_path = "./dataset"
    os.makedirs(final_dataset_path, exist_ok=True)

    # 1. 下载数据集（带进度条）
    if not os.path.exists(save_path):
        print("正在下载CASIA数据集...")
        response = requests.get(download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for data in response.iter_content(1024):
                f.write(data)
                pbar.update(len(data))
        print("下载完成！")

    # 2. 解压数据集
    if not os.path.exists(extract_path):
        print("正在解压数据集...")
        with zipfile.ZipFile(save_path, 'r') as zipf:
            zipf.extractall(extract_path)
        print("解压完成！")

    # 3. 格式转换：转成代码需要的[npy格式]（RGB+掩码+边界+标签）
    if not os.path.exists(os.path.join(final_dataset_path, "rgb_data.npy")):
        print("正在转换数据集格式（适配GB-Net）...")
        raw_path = os.path.join(extract_path, "casia2groundtruth-master")
        real_dir = os.path.join(raw_path, "Au")  # 真实图（未篡改）
        fake_dir = os.path.join(raw_path, "Tp")  # 篡改图
        mask_dir = os.path.join(raw_path, "GT")  # 篡改掩码

        rgb_list, mask_list, labels_list = [], [], []
        img_size = 224

        # 处理真实图（标签0，掩码全0）
        for img_name in os.listdir(real_dir):
            if img_name.endswith(('.jpg','.png')):
                img_path = os.path.join(real_dir, img_name)
                img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
                rgb_arr = np.array(img).transpose(2,0,1)  # [3,224,224]
                mask_arr = np.zeros((1, img_size, img_size))  # 无篡改，掩码全0
                rgb_list.append(rgb_arr)
                mask_list.append(mask_arr)
                labels_list.append(0)  # 未篡改标签0

        # 处理篡改图（标签1，对应掩码）
        for img_name in os.listdir(fake_dir):
            if img_name.endswith(('.jpg','.png')):
                img_path = os.path.join(fake_dir, img_name)
                img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
                rgb_arr = np.array(img).transpose(2,0,1)
                # 匹配对应掩码
                mask_name = img_name.replace('.jpg','_gt.png').replace('.png','_gt.png')
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L').resize((img_size, img_size))
                    mask_arr = np.array(mask)[np.newaxis, :, :] / 255.0  # 归一化到0-1，[1,224,224]
                else:
                    mask_arr = np.zeros((1, img_size, img_size))
                rgb_list.append(rgb_arr)
                mask_list.append(mask_arr)
                labels_list.append(1)  # 篡改标签1

        # 转numpy数组，保存到dataset文件夹
        rgb_data = np.array(rgb_list, dtype=np.float32)
        gt_mask = np.array(mask_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.long)

        # 自动生成边界标签（用BSM的BCM模块，不用手动标）
        bcm = BoundaryCalculationModule()
        gt_bound = bcm(torch.from_numpy(gt_mask)).numpy()

        # 保存npy文件（代码直接读取）
        np.save(os.path.join(final_dataset_path, "rgb_data.npy"), rgb_data)
        np.save(os.path.join(final_dataset_path, "gt_mask.npy"), gt_mask)
        np.save(os.path.join(final_dataset_path, "gt_bound.npy"), gt_bound)
        np.save(os.path.join(final_dataset_path, "labels.npy"), labels)
        print(f"格式转换完成！共{len(rgb_data)}张图，保存在{final_dataset_path}")

    return final_dataset_path

# ===================== 2. 所有核心模块（完整保留，无需修改） =====================
# 限制卷积噪声视图模块
class ConstrainedConvNoise(nn.Module):
    def __init__(self, kernel_size=3, out_channels=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv = nn.Conv2d(3, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self._init_constrained_kernel()

    def _init_constrained_kernel(self):
        init_val = 1.0 / (self.kernel_size**2 - 1)
        nn.init.constant_(self.conv.weight, init_val)
        center = self.kernel_size // 2
        for out_c in range(self.out_channels):
            for in_c in range(3):
                self.conv.weight.data[out_c, in_c, center, center] = -1.0

    def _constrain_kernel(self):
        center = self.kernel_size // 2
        with torch.no_grad():
            for out_c in range(self.out_channels):
                for in_c in range(3):
                    kernel = self.conv.weight.data[out_c, in_c]
                    kernel[center, center] = -1.0
                    non_center = kernel.clone()
                    non_center[center, center] = 0.0
                    sum_non_center = non_center.sum()
                    if sum_non_center != 0:
                        non_center = non_center / sum_non_center
                    kernel = non_center
                    kernel[center, center] = -1.0
                    self.conv.weight.data[out_c, in_c] = kernel

    def forward(self, x):
        if self.training:
            self._constrain_kernel()
        noise_feat = self.conv(x)
        noise_view = x + 0.1 * noise_feat
        return noise_view.clamp(-1, 1)

# GIM+FEM+GEM 全局信息增强模块
class GlobalInformationModule(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        self.SPP = nn.ModuleList([nn.AdaptiveAvgPool2d((1,1)),nn.AdaptiveAvgPool2d((2,2)),nn.AdaptiveAvgPool2d((3,3)),nn.AdaptiveAvgPool2d((6,6))])
        self.encode = nn.ModuleList([nn.Conv2d(c, c//4, 1) for c in in_channels_list])
        self.relu = nn.ReLU()
    def forward(self, multi_scale_feats):
        global_feats = []
        for i, feat in enumerate(multi_scale_feats):
            spp_feats = [pool(feat) for pool in self.SPP]
            spp_feats = torch.cat([F.interpolate(f, size=feat.shape[2:]) for f in spp_feats], dim=1)
            global_feat = self.relu(self.encode[i](spp_feats))
            global_feats.append(global_feat)
        return global_feats

class FeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.non_local = nn.Sequential(nn.Conv2d(in_channels, in_channels//8, 1),nn.ReLU(),nn.Conv2d(in_channels//8, in_channels, 1),nn.Sigmoid())
        self.res_branch = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,padding=1,bias=False),nn.BatchNorm2d(in_channels),nn.ReLU(),nn.Conv2d(in_channels,in_channels,3,padding=1,bias=False),nn.BatchNorm2d(in_channels))
    def forward(self, local_feat, global_feat):
        att_weight = self.non_local(global_feat)
        feat_att = local_feat * att_weight
        return feat_att + self.res_branch(feat_att)

class GlobalInformationEnhancementModule(nn.Module):
    def __init__(self, in_channels_list=[256,512,1024]):
        super().__init__()
        self.GIM = GlobalInformationModule(in_channels_list)
        self.FEMs = nn.ModuleList([FeatureEnhancementModule(c) for c in in_channels_list])
    def forward(self, multi_scale_feats):
        glob_feats = self.GIM(multi_scale_feats)
        return [self.FEMs[i](multi_scale_feats[i], glob_feats[i]) for i in range(len(multi_scale_feats))]

# BCM+BSM 边界监督模块
class BoundaryCalculationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ops = nn.ModuleList([nn.Conv2d(1,1,3,padding=1,bias=False) for _ in range(4)])
        kernels = [torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).reshape(1,1,3,3)/4,
                   torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).reshape(1,1,3,3)/4,
                   torch.tensor([[2,1,0],[1,0,-1],[0,-1,-2]],dtype=torch.float32).reshape(1,1,3,3)/4,
                   torch.tensor([[0,1,2],[-1,0,1],[-2,-1,0]],dtype=torch.float32).reshape(1,1,3,3)/4]
        for i, op in enumerate(self.ops):
            op.weight = nn.Parameter(kernels[i], requires_grad=False)
    def forward(self, pred_mask):
        edges = [op(pred_mask) for op in self.ops]
        edge = torch.sqrt(sum(e**2 for e in edges)) / 2
        return torch.sigmoid(edge)

class BoundarySupervisionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCM = BoundaryCalculationModule()
        self.bound_loss = nn.BCELoss()
    def forward(self, pred_mask, gt_bound):
        pred_bound = self.BCM(pred_mask)
        return pred_bound, self.bound_loss(pred_bound, gt_bound)

# 双残差+裁剪版ResNet50
class DoubleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(),nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False),nn.BatchNorm2d(out_channels))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(),nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False),nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Conv2d(in_channels,out_channels,1,bias=False) if in_channels!=out_channels else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.branch1(x)+self.branch2(x)+self.shortcut(x))

def get_cropped_resnet50():
    resnet = resnet50(pretrained=False)
    return nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)

# 非局部注意力模块
class NonLocalAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.q = nn.Conv2d(channels, channels//reduction,1)
        self.k = nn.Conv2d(channels, channels//reduction,1)
        self.v = nn.Conv2d(channels, channels,1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        B,C,H,W = x.shape
        q,k,v = self.q(x).view(B,-1,H*W).permute(0,2,1),self.k(x).view(B,-1,H*W),self.v(x).view(B,-1,H*W)
        att = F.softmax(torch.bmm(q,k)/np.sqrt(C//reduction),dim=-1)
        return x + self.gamma * torch.bmm(v,att.permute(0,2,1)).view(B,C,H,W)

# 损失函数（多尺度Dice+区分性损失）
def dice_loss(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def global_discriminative_loss(feats, labels, temp=0.07):
    feats_norm = F.normalize(feats, dim=1)
    sim_mat = torch.mm(feats_norm, feats_norm.t())/temp
    label_mask = (labels.unsqueeze(0)==labels.unsqueeze(1)).float() * (1-torch.eye(len(labels),device=labels.device))
    logits = sim_mat - torch.max(sim_mat,dim=1,keepdim=True)[0]
    exp_logits = torch.exp(logits)*label_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1,keepdim=True))
    return -(label_mask*log_prob).sum(dim=1).div(label_mask.sum(dim=1).clamp(min=1e-6)).mean()

# ===================== 3. 完整GB-Net（5关键字全覆盖） =====================
class GB_Net(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.constrained_conv_noise = ConstrainedConvNoise(kernel_size=3, out_channels=3)
        self.rgb_backbone = get_cropped_resnet50()
        self.noise_backbone = get_cropped_resnet50()
        self.rgb_enhance = nn.ModuleList([DoubleResBlock(256,256), DoubleResBlock(512,512), NonLocalAttention(1024)])
        self.noise_enhance = nn.ModuleList([DoubleResBlock(256,256), DoubleResBlock(512,512), NonLocalAttention(1024)])
        self.GEM = GlobalInformationEnhancementModule(in_channels_list=[256,512,1024])
        self.BSM = BoundarySupervisionModule()
        self.mask_decoders = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), nn.Conv2d(128,1,3,padding=1), nn.Sigmoid()),
            nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(), nn.Conv2d(256,1,3,padding=1), nn.Sigmoid()),
            nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.ReLU(), nn.Conv2d(512,1,3,padding=1), nn.Sigmoid())
        ])
        self.global_proj = nn.Linear(256*56*56 + 512*28*28 + 1024*14*14, 256)

    def extract_multi_scale_feat(self, backbone, x, enhance_modules):
        feat = backbone(x)
        feat1 = feat[:, :256, :, :]
        feat2 = feat[:, 256:768, :, :]
        feat3 = feat[:, 768:, :, :]
        feat1 = enhance_modules[0](feat1)
        feat2 = enhance_modules[1](feat2)
        feat3 = enhance_modules[2](feat3)
        return [feat1, feat2, feat3]

    def forward(self, rgb_img):
        noise_img = self.constrained_conv_noise(rgb_img)
        rgb_feats = self.extract_multi_scale_feat(self.rgb_backbone, rgb_img, self.rgb_enhance)
        noise_feats = self.extract_multi_scale_feat(self.noise_backbone, noise_img, self.noise_enhance)
        fuse_feats = [r + n for r, n in zip(rgb_feats, noise_feats)]
        enhance_feats = self.GEM(fuse_feats)
        pred_masks = [decoder(f) for decoder, f in zip(self.mask_decoders, enhance_feats)]
        global_feat = torch.cat([f.flatten(1) for f in enhance_feats], dim=1)
        global_proj_feat = self.global_proj(global_feat)
        return pred_masks, global_proj_feat

# ===================== 4. 数据集类（适配自动下载的npy） =====================
class PixelTamperDataset(Dataset):
    def __init__(self, dataset_path):
        self.rgb = (np.load(os.path.join(dataset_path, "rgb_data.npy"))/255.0 - 0.5)/0.5
        self.gt_mask = np.load(os.path.join(dataset_path, "gt_mask.npy"))
        self.gt_bound = np.load(os.path.join(dataset_path, "gt_bound.npy"))
        self.labels = np.load(os.path.join(dataset_path, "labels.npy"))
    def __len__(self):
        return len(self.rgb)
    def __getitem__(self, idx):
        return self.rgb[idx].astype(np.float32), self.gt_mask[idx].astype(np.float32), self.gt_bound[idx].astype(np.float32), self.labels[idx].astype(np.long)

# ===================== 5. 主训练流程（GitHub一键跑） =====================
if __name__ == "__main__":
    # 自动下载+转换CASIA数据集，返回数据集路径
    dataset_path = download_casia_github()
    # 设备自动适配（GPU/CPU，GitHub Codespaces可自动识别GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    # 初始化模型和训练组件
    model = GB_Net(img_size=224).to(device)
    bsm = BoundarySupervisionModule().to(device)
    optimizer = optim.Adam(list(model.parameters())+list(bsm.parameters()), lr=1e-4, weight_decay=1e-5)
    dataset = PixelTamperDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # 训练配置（论文同款参数）
    num_epochs = 50
    loss_weights = [0.8, 0.15, 0.05] # 掩码损失:边界损失:区分性损失
    print("开始训练GB-Net（5关键字全覆盖+CASIA真实数据）...")

    for epoch in range(num_epochs):
        model.train()
        bsm.train()
        total_loss = 0.0
        for rgb_img, gt_mask, gt_bound, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # 数据移到对应设备
            rgb_img, gt_mask, gt_bound, labels = rgb_img.to(device), gt_mask.to(device), gt_bound.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            pred_masks, global_feat = model(rgb_img)
            # 多尺度掩码/边界损失
            loss_mask = sum(dice_loss(p, gt_mask) for p in pred_masks) / len(pred_masks)
            loss_bound = sum(bsm(p, gt_bound)[1] for p in pred_masks) / len(pred_masks)
            # 全局区分性损失
            loss_disc = global_discriminative_loss(global_feat, labels)
            # 总损失加权
            total_loss_step = loss_weights[0]*loss_mask + loss_weights[1]*loss_bound + loss_weights[2]*loss_disc

            # 反向传播+优化
            total_loss_step.backward()
            optimizer.step()
            total_loss += total_loss_step.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | 平均损失: {avg_loss:.4f} | 掩码损失: {loss_mask.item():.4f}")

    # 保存训练好的模型（GitHub可直接下载）
    torch.save({"model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict()}, "GB-Net_CASIA_trained.pth")
    print("训练完成！模型保存为 GB-Net_CASIA_trained.pth")