import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import os
from PIL import Image

# VGGNet模型定义
class VGGNet(nn.Module):
    def __init__(self, num_classes=10, vgg_type='VGG16'):
        super(VGGNet, self).__init__()
        
        # VGG配置：每个数字表示卷积层的输出通道数，'M'表示MaxPool层
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        
        # 构建特征提取层
        self.features = self._make_layers(cfg[vgg_type])
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 4096),  # 适配CIFAR-10的32x32图像
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # 存储中间特征图和层名称
        self.feature_maps = []
        self.layer_names = []
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        layer_idx = 0
        
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
                layer_idx += 1
                
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 清空特征图和层名称
        self.feature_maps = []
        self.layer_names = []
        
        # 逐层前向传播并保存特征图
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.feature_maps.append(x.clone())
                self.layer_names.append(f'conv_{len(self.feature_maps)}')
        
        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self):
        return self.feature_maps, self.layer_names

# 保存特征热力图函数
def save_feature_maps(model, data_loader, device, save_dir='feature_maps', num_samples=5):
    """
    保存特征热力图到指定目录
    """
    model.eval()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # CIFAR-10类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        for sample_idx, (images, labels) in enumerate(data_loader):
            if sample_idx >= num_samples:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            feature_maps, layer_names = model.get_feature_maps()
            
            # 处理第一个样本
            img = images[0].cpu()
            label = labels[0].cpu().item()
            
            # 反归一化图像用于显示
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            img_display = img * std.view(3, 1, 1) + mean.view(3, 1, 1)
            img_display = torch.clamp(img_display, 0, 1)
            
            # 创建样本专用目录
            sample_dir = os.path.join(save_dir, f'sample_{sample_idx}_{classes[label]}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # 保存原始图像
            plt.figure(figsize=(4, 4))
            plt.imshow(img_display.permute(1, 2, 0))
            plt.title(f'Original Image - {classes[label]}')
            plt.axis('off')
            plt.savefig(os.path.join(sample_dir, '00_original_image.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 保存各层特征图
            for layer_idx, (feature_map, layer_name) in enumerate(zip(feature_maps, layer_names)):
                # 获取第一个样本的前16个通道（如果通道数大于16）
                fmap = feature_map[0].cpu().numpy()
                num_channels = min(16, fmap.shape[0])
                
                # 创建子图网格
                cols = 4
                rows = (num_channels + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
                if rows == 1:
                    axes = axes.reshape(1, -1)
                
                fig.suptitle(f'{layer_name} - Feature Maps (Shape: {fmap.shape})', fontsize=16)
                
                for ch_idx in range(num_channels):
                    row = ch_idx // cols
                    col = ch_idx % cols
                    
                    im = axes[row, col].imshow(fmap[ch_idx], cmap='hot', interpolation='nearest')
                    axes[row, col].set_title(f'Channel {ch_idx}')
                    axes[row, col].axis('off')
                    plt.colorbar(im, ax=axes[row, col])
                
                # 隐藏多余的子图
                for ch_idx in range(num_channels, rows * cols):
                    row = ch_idx // cols
                    col = ch_idx % cols
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, f'{layer_idx+1:02d}_{layer_name}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                # 保存特征图统计信息
                stats_file = os.path.join(sample_dir, f'{layer_idx+1:02d}_{layer_name}_stats.txt')
                with open(stats_file, 'w') as f:
                    f.write(f"Layer: {layer_name}\n")
                    f.write(f"Shape: {fmap.shape}\n")
                    f.write(f"Min: {fmap.min():.4f}\n")
                    f.write(f"Max: {fmap.max():.4f}\n")
                    f.write(f"Mean: {fmap.mean():.4f}\n")
                    f.write(f"Std: {fmap.std():.4f}\n")
            
            print(f"样本 {sample_idx} ({classes[label]}) 的特征图已保存到: {sample_dir}")

# 生成Grad-CAM热力图
def generate_gradcam(model, image, target_class, device, save_path=None):
    model.eval()
    
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # 获取最后一个卷积层
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
    
    # 注册hooks
    handle_backward = target_layer.register_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)
    
    # 前向传播
    image = image.unsqueeze(0).to(device)
    image.requires_grad_()
    
    output = model(image)
    
    # 反向传播
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()
    
    # 获取梯度和激活
    gradients = gradients[0].cpu().data.numpy()[0]
    activations = activations[0].cpu().data.numpy()[0]
    
    # 计算权重
    weights = np.mean(gradients, axis=(1, 2))
    
    # 生成CAM
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32, 32))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    # 保存CAM图像
    if save_path:
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        # 反归一化原图
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        img_display = image.squeeze(0).cpu() * std.view(3, 1, 1) + mean.view(3, 1, 1)
        img_display = torch.clamp(img_display, 0, 1)
        plt.imshow(img_display.permute(1, 2, 0))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_display.permute(1, 2, 0))
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title('Grad-CAM')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # 清理hooks
    handle_backward.remove()
    handle_forward.remove()
    
    return cam

# 主训练函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir='./tensorboard_logs/vggnet_cifar10')
    
    # 超参数
    batch_size = 64  # VGG较大，使用较小的batch size
    learning_rate = 0.0001
    num_epochs = 30
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载CIFAR-10数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 创建模型
    model = VGGNet(num_classes=10, vgg_type='VGG16').to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 将模型图写入TensorBoard
    sample_input = torch.randn(1, 3, 32, 32).to(device)
    writer.add_graph(model, sample_input)
    
    # 训练循环
    global_step = 0
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # 记录到TensorBoard
            writer.add_scalar('Training/Loss', loss.item(), global_step)
            writer.add_scalar('Training/Accuracy', 100 * correct_train / total_train, global_step)
            
            global_step += 1
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Train Acc: {100 * correct_train / total_train:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 每个epoch结束后评估模型
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        # 计算平均值
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        
        # 记录到TensorBoard
        writer.add_scalar('Test/Loss', avg_test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], 测试准确率: {test_accuracy:.2f}%')
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'vggnet_best.pth')
            print(f"保存最佳模型，准确率: {best_accuracy:.2f}%")
        
        # 每10个epoch保存特征图
        if (epoch + 1) % 10 == 0:
            print("保存特征图...")
            save_feature_maps(model, test_loader, device, 
                            save_dir=f'./vgg_feature_png/vgg_feature_maps_epoch_{epoch+1}', 
                            num_samples=3)
            
            # 生成几个样本的Grad-CAM
            gradcam_dir = f'./vgg_feature_png/vgg_gradcam_epoch_{epoch+1}'
            os.makedirs(gradcam_dir, exist_ok=True)
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    if i >= 3:  # 只处理3个样本
                        break
                    images, labels = images.to(device), labels.to(device)
                    
                    for j in range(min(2, images.size(0))):  # 每个batch取2个样本
                        img = images[j]
                        label = labels[j].item()
                        
                        cam_path = os.path.join(gradcam_dir, f'gradcam_sample_{i}_{j}_class_{label}.png')
                        generate_gradcam(model, img, label, device, cam_path)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'vggnet_cifar10_final.pth')
    print('最终模型已保存为 vggnet_cifar10_final.pth')
    
    # 关闭TensorBoard writer
    writer.close()
    
    print("\n训练完成！")
    print("使用以下命令启动TensorBoard查看训练过程:")
    print("tensorboard --logdir=runs")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()