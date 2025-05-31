import torch

print("=== GPU 详细信息 ===")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}:")
        print(f"  名称: {props.name}")
        print(f"  显存: {props.total_memory / 1024**3:.2f}GB")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  多处理器数量: {props.multi_processor_count}")
        
    print(f"\n当前默认GPU: {torch.cuda.current_device()}")
    print(f"当前GPU名称: {torch.cuda.get_device_name()}")
    
    # 测试GPU计算
    print("\n=== GPU计算测试 ===")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"计算完成，结果形状: {z.shape}")
    print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    
    # 检查是否真的在使用独显
    print(f"\n当前设备属性:")
    current_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"  设备名称: {current_props.name}")
    print(f"  总显存: {current_props.total_memory / 1024**3:.2f}GB")