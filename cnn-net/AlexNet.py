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
import cv2
from PIL import Image

# AlexNet模型定义
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 输入通道数为3，输出通道数为64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 输入通道数为64，输出通道数为192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 256, kernel_size=3, padding=1),  # 输入通道数为192，输出通道数为256
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 输入通道数为256，输出通道数为256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 1024),  # 输入特征数为256*2*2，输出特征数为1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),  # 输入特征数为1024，输出特征数为1024
        )

        # 存储中间特征图
        self.feature_maps = []
    
    def forward(self, x):
        # 清空中间特征图
        self.feature_maps = []

        # 前向传播
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.feature_maps.append(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self):
        """获取中间特征图"""
        return self.feature_maps

# 特征热力图
def visualize_feature_maps(model, data_loader, device, num_samples=5):
    model.eval()
    
    # CIFAR-10数据集的类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if i >= num_samples:
                break

            images = images.to(device)
            labels = labels.to(device)

            # 前向传播获取特征图
            outputs = model(images)
            feature_maps = model.get_feature_maps()

            # 只处理第一个样本的特征图
            img = images[0].cpu()
            label = labels[0].cpu().item()

            # 反归一化图片用于显示
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            img_display = img * std.view(3, 1, 1) + mean.view(3, 1, 1)
            img_display = torch.clamp(img_display, 0, 1)

            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Label: {classes[label]}', fontsize=16)

            # 显示原始图片
            axes[0, 0].imshow(img_display.permute(1, 2, 0))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            # 显示特征图
            for idx, feature_map in enumerate(feature_maps):
                if idx >= 5:
                    break

                fmap = feature_map[0, 0].cpu().numpy()  # 获取第一个样本的第一个通道特征图
                
                row = (idx + 1) // 3
                col = (idx + 1) % 3

                im = axes[row, col].imshow(fmap, cmap='hot', interpolation='nearest')
                axes[row, col].set_title(f'Feature Map {idx + 1}')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col])
            
            plt.tight_layout()
            plt.savefig(f'feature_maps_sample_{i}.png', dpi = 150, bbox_inches='tight')
            plt.show()

            break  # 只处理一个batch的第一个样本

        # 生成Grad-CAM热力图
def generate_gradcam(model, image, target_class, device):
    model.eval()
    
    # 注册hook获取梯度和特征图
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
    cam = cam / np.max(cam)
    
    # 清理hooks
    handle_backward.remove()
    handle_forward.remove()
    
    return cam

# 主训练函数
def main():
    # 设置设备为 mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建 tensorboard writer
    writer = SummaryWriter(log_dir='./tensorboard_logs/alexnet_cifar10')

    # 超参数
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 20

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # 加载CIFAR-10数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 初始化模型
    model = AlexNet(num_classes=10).to(device)
    
    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 将模型图写入TensorBoard
    sample_input = torch.randn(1, 3, 32, 32).to(device)
    writer.add_graph(model, sample_input)

    # 训练模型
    global_step = 0

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

            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # 写入TensorBoard
            writer.add_scalar('Training/Loss', loss.item(), global_step)
            writer.add_scalar('Training/Accuracy', 100 * correct_train / total_train, global_step)
            global_step += 1

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/ {num_epochs}], Step [{i+1}/{len(train_loader)}],'
                      f'Loss: {loss.item():.4f}, Train Acc: {100 * correct_train / total_train:.2f}%')
        
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
        
        # 记录学习率
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], 测试准确率: {test_accuracy:.2f}%')
        
        # 每5个epoch生成特征图可视化
        if (epoch + 1) % 5 == 0:
            print("生成特征图可视化...")
            visualize_feature_maps(model, test_loader, device, num_samples=2)
    
    # 保存模型
    torch.save(model.state_dict(), 'alexnet_cifar10_with_viz.pth')
    print('模型已保存为 alexnet_cifar10_with_viz.pth')
    
    # 关闭TensorBoard writer
    writer.close()
    
    print("\n训练完成！")
    print("使用以下命令启动TensorBoard查看训练过程:")
    print("tensorboard --logdir=runs")

if __name__ == '__main__':
    main()     

