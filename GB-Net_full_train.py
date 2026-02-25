import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from torchvision.models import resnet50
import os
import requests
import zipfile
from PIL import Image
from tqdm import tqdm
import random
from torchvision import transforms

# ===================== 1. 改进的数据集下载和预处理 =====================
def download_casia_github():
    """改进的CASIA数据集下载和预处理函数"""
    
    # 使用更可靠的下载源（CASIA2数据集）
    # 方案A: 使用GitHub上的CASIA2镜像
    download_urls = [
        "https://github.com/namtpham/casia2groundtruth/archive/refs/heads/master.zip",
        "https://github.com/selimsef/detecting-spliced-faces/raw/master/data/CASIA2.zip"  # 备用源
    ]
    
    save_path = "./casia_dataset.zip"
    extract_path = "./casia_raw"
    final_dataset_path = "./dataset"
    os.makedirs(final_dataset_path, exist_ok=True)
    
    # 检查是否已经处理过
    if os.path.exists(os.path.join(final_dataset_path, "rgb_data.npy")):
        print(f"数据集已存在：{final_dataset_path}")
        return final_dataset_path
    
    # 尝试下载
    downloaded = False
    for url in download_urls:
        try:
            print(f"尝试从 {url} 下载...")
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                with open(save_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        f.write(data)
                        pbar.update(len(data))
                print("下载完成！")
                downloaded = True
                break
        except Exception as e:
            print(f"下载失败：{e}")
            continue
    
    if not downloaded:
        # 如果下载失败，提示用户手动下载
        print("""
        =================================================
        自动下载失败。请手动下载CASIA数据集：
        
        方法1：从百度网盘下载（提取码：abcd）
        链接：https://pan.baidu.com/s/1xxxxxx
        
        方法2：从Google Drive下载
        链接：https://drive.google.com/file/d/xxxxxx
        
        下载后解压到 ./casia_raw 文件夹，然后重新运行程序。
        =================================================
        """)
        raise Exception("数据集下载失败，请手动下载")
    
    # 解压数据集
    print("正在解压数据集...")
    with zipfile.ZipFile(save_path, 'r') as zipf:
        zipf.extractall(extract_path)
    print("解压完成！")
    
    # 格式转换
    print("正在转换数据集格式...")
    
    # 查找正确的目录结构
    base_dir = extract_path
    for root, dirs, files in os.walk(extract_path):
        if 'Au' in dirs and 'Tp' in dirs and 'GT' in dirs:
            base_dir = root
            break
    
    real_dir = os.path.join(base_dir, "Au")  # 真实图
    fake_dir = os.path.join(base_dir, "Tp")  # 篡改图
    mask_dir = os.path.join(base_dir, "GT")  # 掩码
    
    # 检查目录是否存在
    if not os.path.exists(real_dir):
        # 尝试其他可能的目录名
        real_dir = os.path.join(base_dir, "Authentic")
        fake_dir = os.path.join(base_dir, "Tampered")
        mask_dir = os.path.join(base_dir, "Mask")
    
    # 收集所有图像
    rgb_list, mask_list, labels_list = [], [], []
    img_size = 224
    
    # 处理真实图像
    print("处理真实图像...")
    real_images = []
    for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
        real_images.extend(glob.glob(os.path.join(real_dir, f'*{ext}')))
        real_images.extend(glob.glob(os.path.join(real_dir, f'*{ext.upper()}')))
    
    for img_path in tqdm(real_images[:500]):  # 限制数量避免内存溢出
        try:
            img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
            rgb_arr = np.array(img).transpose(2, 0, 1)  # [3,224,224]
            mask_arr = np.zeros((1, img_size, img_size))
            rgb_list.append(rgb_arr)
            mask_list.append(mask_arr)
            labels_list.append(0)
        except Exception as e:
            print(f"处理 {img_path} 时出错：{e}")
    
    # 处理篡改图像
    print("处理篡改图像...")
    fake_images = []
    for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
        fake_images.extend(glob.glob(os.path.join(fake_dir, f'*{ext}')))
    
    for img_path in tqdm(fake_images[:500]):  # 限制数量
        try:
            img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
            rgb_arr = np.array(img).transpose(2, 0, 1)
            
            # 匹配掩码（尝试多种命名规则）
            base_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            possible_masks = [
                os.path.join(mask_dir, f"{name_without_ext}_gt.png"),
                os.path.join(mask_dir, f"{name_without_ext}.png"),
                os.path.join(mask_dir, base_name.replace('.jpg', '_gt.png').replace('.png', '_gt.png')),
                os.path.join(mask_dir, base_name)
            ]
            
            mask_path = None
            for pm in possible_masks:
                if os.path.exists(pm):
                    mask_path = pm
                    break
            
            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L').resize((img_size, img_size))
                mask_arr = np.array(mask)[np.newaxis, :, :] / 255.0
                # 二值化掩码
                mask_arr = (mask_arr > 0.5).astype(np.float32)
            else:
                print(f"未找到掩码：{img_path}")
                mask_arr = np.zeros((1, img_size, img_size))
            
            rgb_list.append(rgb_arr)
            mask_list.append(mask_arr)
            labels_list.append(1)
            
        except Exception as e:
            print(f"处理 {img_path} 时出错：{e}")
    
    # 确保数据平衡
    print(f"真实图像：{labels_list.count(0)}，篡改图像：{labels_list.count(1)}")
    
    # 转换为numpy数组
    rgb_data = np.array(rgb_list, dtype=np.float32)
    gt_mask = np.array(mask_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    
    # 检查数据
    print(f"RGB数据形状：{rgb_data.shape}")
    print(f"掩码数据形状：{gt_mask.shape}")
    print(f"标签形状：{labels.shape}")
    
    # 生成边界标签
    print("生成边界标签...")
    bcm = BoundaryCalculationModule()
    gt_bound = bcm(torch.from_numpy(gt_mask)).numpy()
    
    # 保存数据
    np.save(os.path.join(final_dataset_path, "rgb_data.npy"), rgb_data)
    np.save(os.path.join(final_dataset_path, "gt_mask.npy"), gt_mask)
    np.save(os.path.join(final_dataset_path, "gt_bound.npy"), gt_bound)
    np.save(os.path.join(final_dataset_path, "labels.npy"), labels)
    
    print(f"数据集保存完成！共 {len(rgb_data)} 张图像")
    return final_dataset_path

# ===================== 2. 改进的数据集类（支持动态加载） =====================
class ImprovedPixelTamperDataset(Dataset):
    """改进的数据集类，支持动态加载和增强"""
    
    def __init__(self, dataset_path, transform=None, max_samples=None):
        """
        Args:
            dataset_path: 数据集路径
            transform: 数据增强
            max_samples: 最大样本数（用于调试）
        """
        self.dataset_path = dataset_path
        
        # 加载数据（如果文件太大，可以考虑mmap模式）
        self.rgb = np.load(os.path.join(dataset_path, "rgb_data.npy"), mmap_mode='r')
        self.gt_mask = np.load(os.path.join(dataset_path, "gt_mask.npy"), mmap_mode='r')
        self.gt_bound = np.load(os.path.join(dataset_path, "gt_bound.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(dataset_path, "labels.npy"), mmap_mode='r')
        
        if max_samples:
            self.rgb = self.rgb[:max_samples]
            self.gt_mask = self.gt_mask[:max_samples]
            self.gt_bound = self.gt_bound[:max_samples]
            self.labels = self.labels[:max_samples]
        
        # 数据增强
        self.transform = transform or self._get_default_transform()
        
        # 检查数据有效性
        self._validate_data()
    
    def _get_default_transform(self):
        """默认的数据增强"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _validate_data(self):
        """验证数据的有效性"""
        print(f"数据集验证：")
        print(f"  RGB数据范围：[{self.rgb.min():.3f}, {self.rgb.max():.3f}]")
        print(f"  掩码数据范围：[{self.gt_mask.min():.3f}, {self.gt_mask.max():.3f}]")
        print(f"  标签分布：0类 {np.sum(self.labels==0)}，1类 {np.sum(self.labels==1)}")
        
        # 检查无效数据
        invalid_mask = np.isnan(self.rgb).any() or np.isinf(self.rgb).any()
        if invalid_mask:
            print("警告：RGB数据包含NaN或Inf值")
    
    def __len__(self):
        return len(self.rgb)
    
    def __getitem__(self, idx):
        # 获取数据
        rgb = self.rgb[idx].copy()  # 复制避免影响mmap
        mask = self.gt_mask[idx].copy()
        bound = self.gt_bound[idx].copy()
        label = self.labels[idx]
        
        # 归一化RGB（如果还没归一化）
        if rgb.max() > 10:  # 假设是[0,255]范围
            rgb = rgb / 255.0
        
        # 标准化
        rgb = (rgb - 0.5) / 0.5
        
        # 转换为tensor
        rgb = torch.from_numpy(rgb).float()
        mask = torch.from_numpy(mask).float()
        bound = torch.from_numpy(bound).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return rgb, mask, bound, label

# ===================== 3. 数据增强函数 =====================
class DataAugmentation:
    """数据增强类"""
    
    @staticmethod
    def random_flip(rgb, mask, bound):
        """随机翻转"""
        if random.random() > 0.5:
            rgb = torch.flip(rgb, dims=[2])
            mask = torch.flip(mask, dims=[2])
            bound = torch.flip(bound, dims=[2])
        return rgb, mask, bound
    
    @staticmethod
    def random_rotation(rgb, mask, bound):
        """随机旋转"""
        k = random.randint(0, 3)
        if k > 0:
            rgb = torch.rot90(rgb, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])
            bound = torch.rot90(bound, k, dims=[1, 2])
        return rgb, mask, bound
    
    @staticmethod
    def random_noise(rgb):
        """添加随机噪声"""
        if random.random() > 0.3:
            noise = torch.randn_like(rgb) * 0.01
            rgb = rgb + noise
        return rgb

# ===================== 4. 改进的训练流程 =====================
def train_with_improved_dataset():
    """改进的训练流程"""
    
    # 自动下载数据集
    try:
        dataset_path = download_casia_github()
    except Exception as e:
        print(f"数据集准备失败：{e}")
        print("请检查网络连接或手动下载数据集")
        return
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # 创建数据集
    full_dataset = ImprovedPixelTamperDataset(dataset_path, max_samples=1000)  # 限制样本数用于测试
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,  # 减小batch size
        shuffle=True, 
        num_workers=0,  # 避免多进程问题
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=0
    )
    
    # 初始化模型
    model = GB_Net(img_size=224).to(device)
    bsm = BoundarySupervisionModule().to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(bsm.parameters()), lr=1e-4)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 训练参数
    num_epochs = 50
    loss_weights = [0.8, 0.15, 0.05]
    
    print("开始训练...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        bsm.train()
        train_loss = 0.0
        
        for batch_idx, (rgb, mask, bound, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # 数据增强
            rgb, mask, bound = DataAugmentation.random_flip(rgb, mask, bound)
            rgb, mask, bound = DataAugmentation.random_rotation(rgb, mask, bound)
            rgb = DataAugmentation.random_noise(rgb)
            
            # 移动到设备
            rgb = rgb.to(device)
            mask = mask.to(device)
            bound = bound.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            pred_masks, global_feat = model(rgb)
            
            # 计算损失
            loss_mask = sum(dice_loss(p, mask) for p in pred_masks) / len(pred_masks)
            
            loss_bound = 0
            for p in pred_masks:
                pred_bound, bound_loss = bsm(p, bound)
                loss_bound += bound_loss
            loss_bound /= len(pred_masks)
            
            loss_disc = global_discriminative_loss(global_feat, labels)
            
            # 总损失
            loss = loss_weights[0] * loss_mask + loss_weights[1] * loss_bound + loss_weights[2] * loss_disc
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # 打印批次信息
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss={loss.item():.4f}, Mask={loss_mask.item():.4f}, Bound={loss_bound.item():.4f}, Disc={loss_disc.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb, mask, bound, labels in val_loader:
                rgb = rgb.to(device)
                mask = mask.to(device)
                bound = bound.to(device)
                labels = labels.to(device)
                
                pred_masks, global_feat = model(rgb)
                
                loss_mask = sum(dice_loss(p, mask) for p in pred_masks) / len(pred_masks)
                loss_bound = sum(bsm(p, bound)[1] for p in pred_masks) / len(pred_masks)
                loss_disc = global_discriminative_loss(global_feat, labels)
                
                loss = loss_weights[0] * loss_mask + loss_weights[1] * loss_bound + loss_weights[2] * loss_disc
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # 调整学习率
        scheduler.step(avg_val_loss)
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, f"checkpoint_epoch_{epoch+1}.pth")
    
    # 保存最终模型
    torch.save(model.state_dict(), "GB-Net_CASIA_final.pth")
    print("训练完成！")

# ===================== 5. 诊断工具 =====================
def diagnose_dataset(dataset_path):
    """诊断数据集问题"""
    
    print("=" * 50)
    print("数据集诊断")
    print("=" * 50)
    
    # 检查文件是否存在
    required_files = ["rgb_data.npy", "gt_mask.npy", "gt_bound.npy", "labels.npy"]
    for file in required_files:
        file_path = os.path.join(dataset_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ {file}: {size:.2f} MB")
        else:
            print(f"✗ {file}: 不存在")
    
    # 检查数据内容
    try:
        rgb = np.load(os.path.join(dataset_path, "rgb_data.npy"), mmap_mode='r')
        mask = np.load(os.path.join(dataset_path, "gt_mask.npy"), mmap_mode='r')
        labels = np.load(os.path.join(dataset_path, "labels.npy"), mmap_mode='r')
        
        print(f"\n数据统计：")
        print(f"  样本总数：{len(rgb)}")
        print(f"  RGB形状：{rgb.shape}")
        print(f"  掩码形状：{mask.shape}")
        print(f"  标签分布：0类 {np.sum(labels==0)}，1类 {np.sum(labels==1)}")
        
        # 检查掩码有效性
        mask_sum = mask.sum(axis=(1,2,3))
        print(f"  掩码统计：均值 {mask_sum.mean():.2f}，最大 {mask_sum.max():.2f}")
        
        # 检查数据范围
        print(f"  RGB范围：[{rgb.min():.3f}, {rgb.max():.3f}]")
        
    except Exception as e:
        print(f"数据加载失败：{e}")

if __name__ == "__main__":
    # 运行诊断
    if os.path.exists("./dataset"):
        diagnose_dataset("./dataset")
    
    # 运行训练
    train_with_improved_dataset()
