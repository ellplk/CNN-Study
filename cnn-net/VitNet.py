import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import gc
import os

# RTX 4060 8GB 优化版ViT
class ViTCIFAR10_RTX4060(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ViTCIFAR10_RTX4060, self).__init__()
        
        # 加载预训练的ViT模型
        if pretrained:
            weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.vit = torchvision.models.vit_b_16(weights=weights)
        
        # 获取预训练模型期望的输入尺寸
        self.vit_expected_input_size = self.vit.image_size
        
        # 修改分类头
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, num_classes)
        
    def forward(self, x):
        # 动态调整输入尺寸到ViT期望的224x224
        current_h, current_w = x.shape[-2:]
        expected_size = self.vit_expected_input_size

        if current_h != expected_size or current_w != expected_size:
            x = torch.nn.functional.interpolate(x, 
                                              size=(expected_size, expected_size),
                                              mode='bilinear', align_corners=False)
        
        result = self.vit(x)
        return result

def check_gpu_setup():
    """检查GPU配置并选择RTX 4060"""
    print("=== RTX 4060 GPU设置检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        # 查找RTX 4060
        rtx4060_device = None
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  显存: {gpu_props.total_memory / 1024**3:.2f}GB")
            print(f"  计算能力: {gpu_props.major}.{gpu_props.minor}")
            
            # 查找RTX 4060
            if "4060" in gpu_props.name or "RTX 4060" in gpu_props.name:
                rtx4060_device = i
                print(f"  ✅ 找到RTX 4060!")
        
        if rtx4060_device is not None:
            # 使用RTX 4060
            torch.cuda.set_device(rtx4060_device)
            print(f"设置使用GPU {rtx4060_device}: {torch.cuda.get_device_properties(rtx4060_device).name}")
            device = torch.device(f'cuda:{rtx4060_device}')
        else:
            # 如果没找到RTX 4060，使用最后一个GPU（通常是独显）
            rtx4060_device = torch.cuda.device_count() - 1
            torch.cuda.set_device(rtx4060_device)
            print(f"未找到RTX 4060，使用GPU {rtx4060_device}: {torch.cuda.get_device_properties(rtx4060_device).name}")
            device = torch.device(f'cuda:{rtx4060_device}')
        
        print(f"当前使用GPU: {torch.cuda.current_device()}")
        
        # 清空显存缓存
        torch.cuda.empty_cache()
        print("已清空GPU缓存")
        
        return device
    else:
        print("CUDA不可用，程序将退出")
        exit(1)

def main():
    # 检查GPU设置并获取正确的设备
    device = check_gpu_setup()
    print("=" * 35)
    print(f'使用设备: {device}')
    
    # RTX 4060 8GB 优化参数
    batch_size = 64         # 针对8GB显存优化的batch size
    learning_rate = 3e-4    # 适合ViT的学习率
    num_epochs = 25
    input_size = 160        # 初始输入尺寸，会自动调整到224
    
    # 数据预处理 - ImageNet预训练权重的标准化参数
    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载CIFAR-10数据集
    print("加载CIFAR-10数据集...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # 优化的DataLoader设置
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=6,          # 充分利用CPU
        pin_memory=True,        # 加速GPU传输
        persistent_workers=True, # 减少worker重启开销
        prefetch_factor=3       # 预取数据
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,  # 测试时使用相同batch size
        shuffle=False, 
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")
    print(f"训练批次: {len(train_loader)}, 测试批次: {len(test_loader)}")
    
    # 创建模型
    print("创建ViT模型...")
    model = ViTCIFAR10_RTX4060(num_classes=10, pretrained=True).to(device)
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器 - Warmup + Cosine
    warmup_epochs = 3
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs)
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    
    # 训练状态
    best_accuracy = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"\n开始训练 - {num_epochs} epochs")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # === 训练阶段 ===
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Warmup学习率
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_scale
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # 混合精度前向传播
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # 显存管理
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
                
            # 进度显示
            if (i + 1) % 150 == 0:
                current_acc = 100 * correct_train / total_train
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f'Epoch [{epoch+1:2d}/{num_epochs}] '
                      f'Step [{i+1:4d}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {current_acc:.2f}% '
                      f'GPU: {allocated:.2f}/{reserved:.2f}GB '
                      f'Device: {device}')
        
        # 更新学习率
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # 计算训练准确率
        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # === 测试阶段 ===
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        avg_test_loss = test_loss / len(test_loader)
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        # 显示结果
        current_lr = optimizer.param_groups[0]['lr']
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        print(f'\nEpoch [{epoch+1:2d}/{num_epochs}] 完成')
        print(f'训练 - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%')
        print(f'测试 - Loss: {avg_test_loss:.4f}, Acc: {test_accuracy:.2f}%')
        print(f'学习率: {current_lr:.6f}')
        print(f'显存使用: {allocated:.2f}GB / {reserved:.2f}GB (GPU: {device})')
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies
            }, 'vit_rtx4060_best.pth')
            print(f'💾 保存最佳模型! 准确率: {best_accuracy:.2f}%')
        
        print('-' * 60)
        
        # 显存清理
        torch.cuda.empty_cache()
        gc.collect()
    
    # 训练完成
    print(f'\n🎉 训练完成!')
    print(f'📊 最佳测试准确率: {best_accuracy:.2f}%')
    print(f'📁 最佳模型已保存为: vit_rtx4060_best.pth')
    
    # 最终显存清理
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    # 移除CUDA_LAUNCH_BLOCKING以提高性能
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 注释掉以提高性能
    
    # 启用cuDNN优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    print("🚀 RTX 4060 8GB ViT-CIFAR10 训练程序")
    print("=" * 50)
    
    main()