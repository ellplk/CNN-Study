import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import gc

# 内存优化版ViT
class ViTCIFAR10_Optimized(nn.Module):
    def __init__(self, num_classes=10, pretrained=True): # 移除了 input_size 参数
        super(ViTCIFAR10_Optimized, self).__init__()
        
        # 加载预训练的ViT模型，使用 'weights' 参数
        if pretrained:
            weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.vit = torchvision.models.vit_b_16(weights=weights)
        
        # 获取预训练模型期望的输入尺寸 (例如 224 for vit_b_16)
        self.vit_expected_input_size = self.vit.image_size
        
        # 修改分类头
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, num_classes)
        
    def forward(self, x):
        # x 来自 DataLoader, 可能尺寸是 128x128 或 96x96 等
        # 在这里，我们将 x 调整到 ViT 预训练模型期望的尺寸
        current_h, current_w = x.shape[-2:]
        expected_size = self.vit_expected_input_size

        if current_h != expected_size or current_w != expected_size:
            x = torch.nn.functional.interpolate(x, 
                                              size=(expected_size, expected_size),
                                              mode='bilinear', align_corners=False)
        
        # 现在 x 的尺寸是 (B, 3, expected_size, expected_size)，例如 (B, 3, 224, 224)
        # ViT 模型内部的 image_size 属性不需要修改，它将正确处理输入
        result = self.vit(x)
        return result

# 内存优化版主函数
def main():
    # 设备设置
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 优化后的超参数
    batch_size = 32         # 减小batch size
    learning_rate = 0.0001
    num_epochs = 20
    input_size = 128        # DataLoader 输出的图像尺寸 (用于初步的内存优化)
    
    # 数据预处理 - 使用较小的输入尺寸
    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # 降低分辨率到 input_size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet 均值和标准差
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # 使用pin_memory和num_workers优化
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True if device.type == 'cuda' else False) # pin_memory 在 MPS 上无效
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # 创建优化后的模型
    model = ViTCIFAR10_Optimized(num_classes=10, pretrained=True).to(device) # 不再传递 input_size
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 开启混合精度训练（如果支持CUDA）
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True if device.type == 'cuda' else False), \
                             labels.to(device, non_blocking=True if device.type == 'cuda' else False)
            
            # 清零梯度
            optimizer.zero_grad(set_to_none=True) # 更高效的梯度清零
            
            # 混合精度训练
            if scaler is not None: # 仅当使用 CUDA 且 scaler 有效时
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # 在 clip_grad_norm_ 之前 unscale
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通训练 (适用于 CPU 或 MPS)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # 定期清理内存 (主要对 CUDA 有效)
            if (i + 1) % 50 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect() # Python 垃圾回收
            
            # 每100步打印一次
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # 计算训练准确率
        train_accuracy = 100 * correct_train / total_train
        
        # 测试阶段
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True if device.type == 'cuda' else False), \
                                 labels.to(device, non_blocking=True if device.type == 'cuda' else False)
                
                if scaler is not None: # 同样适用于测试时的 autocast，尽管通常不是必须的
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                    
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'训练准确率: {train_accuracy:.2f}%, 测试准确率: {test_accuracy:.2f}%')
        print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 显示内存使用情况
        if device.type == 'cuda':
            print(f'GPU内存 Allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB / Reserved: {torch.cuda.memory_reserved(device)/1024**3:.2f}GB')
        elif device.type == 'mps':
            # torch.mps.current_allocated_memory() 和 .driver_allocated_memory() 可以使用
            print(f'MPS内存 Current Allocated: {torch.mps.current_allocated_memory()/1024**3:.2f}GB')
        
        print('-' * 50)
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'vit_best_optimized.pth')
            print(f'保存最佳模型，准确率: {best_accuracy:.2f}%')
        
        # 清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f'\n训练完成！最佳准确率: {best_accuracy:.2f}%')

# 极简版本 - 最小内存占用
def main_minimal():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 极简参数
    batch_size = 16      # 非常小的batch
    input_size_dl = 96   # DataLoader 输出的图像尺寸
    num_epochs_minimal = 10 # 减少 epoch 数
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((input_size_dl, input_size_dl)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 数据集
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 模型
    model = ViTCIFAR10_Optimized(num_classes=10, pretrained=True).to(device) # 使用更新后的类
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001) # 保持 AdamW
    
    # 简化训练循环
    for epoch in range(num_epochs_minimal):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # 为了极简，可以减少训练步骤，但这里我们还是完整遍历
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 50 == 0: # 减少打印频率
                print(f'Minimal - Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}')
        
        # 简单测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Minimal - Epoch {epoch+1}, 准确率: {accuracy:.2f}%')
        if device.type == 'mps':
            print(f'MPS内存 Current Allocated: {torch.mps.current_allocated_memory()/1024**3:.2f}GB')
        gc.collect()

if __name__ == '__main__':
    # 选择运行模式
    print("选择运行模式:")
    print("1. 优化版本 (推荐)")
    print("2. 极简版本 (最低内存)")
    
    choice = input("请输入选择 (1 or 2): ")
    
    if choice == "2":
        main_minimal()
    else:
        main()