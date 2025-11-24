#!/usr/bin/env python3
"""
测试脚本：验证自动内存检测功能
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_providers import LocalLLModel
import platform
import psutil

def test_memory_detection():
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
        print(f"\nCUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    总内存: {props.total_memory / (1024**3):.2f} GB")
        
        print(f"MPS 可用: {torch.backends.mps.is_available()}")
    except Exception as e:
        print(f"GPU 检测错误: {e}")
    
    print("\n" + "=" * 60)
    print("自动检测的内存配置")
    print("=" * 60)
    
    # 获取自动检测的内存配置
    max_memory = LocalLLModel.get_available_memory()
    print(f"检测到的内存配置: {max_memory}")
    
    # 解释配置
    print("\n配置说明:")
    for key, value in max_memory.items():
        if key == "cpu":
            print(f"  CPU 内存: {value}")
        else:
            print(f"  GPU {key} 内存: {value}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_memory_detection()
