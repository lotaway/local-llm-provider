#!/usr/bin/env python3
"""
简化测试脚本：仅测试内存检测逻辑
"""
import platform
import psutil

def get_available_memory():
    """
    自动检测系统可用内存配置
    返回格式: {0: "20GiB", "cpu": "60GiB"}
    """
    max_memory = {}
    system = platform.system()
    
    # 检测是否为 Mac 系统
    is_mac = system == "Darwin"
    
    if is_mac:
        # Mac 系统：使用 MPS (Metal Performance Shaders) 或 CPU
        # 获取系统总内存
        total_memory = psutil.virtual_memory().total
        # 预留 8GB 给系统，其余可用于模型
        available_memory_gb = max(1, (total_memory / (1024**3)) - 8)
        max_memory["cpu"] = f"{int(available_memory_gb)}GiB"
        
        # Mac 不使用 CUDA，如果有 MPS 可以使用
        try:
            import torch
            if torch.backends.mps.is_available():
                # MPS 设备共享系统内存，这里不单独设置
                print("检测到 MPS 支持")
        except:
            pass
    else:
        # Windows/Linux 系统
        # 检测 CUDA GPU
        try:
            import torch
            if torch.cuda.is_available():
                # 获取第一个 GPU 的可用内存
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # 预留 2GB 给系统，其余可用于模型
                available_gpu_gb = max(1, (gpu_memory / (1024**3)) - 2)
                max_memory[0] = f"{int(available_gpu_gb)}GiB"
        except:
            pass
        
        # 获取系统 CPU 内存
        total_memory = psutil.virtual_memory().total
        # 预留 8GB 给系统
        available_cpu_gb = max(1, (total_memory / (1024**3)) - 8)
        max_memory["cpu"] = f"{int(available_cpu_gb)}GiB"
    
    return max_memory

def main():
    print("=" * 60)
    print("系统信息检测")
    print("=" * 60)
    
    # 显示系统信息
    print(f"操作系统: {platform.system()}")
    print(f"平台: {platform.platform()}")
    print(f"处理器: {platform.processor()}")
    
    # 显示内存信息
    mem = psutil.virtual_memory()
    print(f"\n系统总内存: {mem.total / (1024**3):.2f} GB")
    print(f"可用内存: {mem.available / (1024**3):.2f} GB")
    print(f"已用内存: {mem.used / (1024**3):.2f} GB")
    print(f"内存使用率: {mem.percent}%")
    
    # 检测 GPU
    try:
        import torch
        print(f"\nPyTorch 已安装")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    总内存: {props.total_memory / (1024**3):.2f} GB")
        
        if hasattr(torch.backends, 'mps'):
            print(f"MPS 可用: {torch.backends.mps.is_available()}")
    except ImportError:
        print("\nPyTorch 未安装，跳过 GPU 检测")
    except Exception as e:
        print(f"\nGPU 检测错误: {e}")
    
    print("\n" + "=" * 60)
    print("自动检测的内存配置")
    print("=" * 60)
    
    # 获取自动检测的内存配置
    max_memory = get_available_memory()
    print(f"检测到的内存配置: {max_memory}")
    
    # 解释配置
    print("\n配置说明:")
    for key, value in max_memory.items():
        if key == "cpu":
            print(f"  CPU 内存: {value} (系统总内存减去 8GB 预留)")
        else:
            print(f"  GPU {key} 内存: {value} (GPU 总内存减去 2GB 预留)")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
