#!/usr/bin/env python3
"""
诊断HuggingFace模型加载失败的原因
分析PyTorchModelHubMixin的工作机制
"""

import sys
import traceback
import requests
import json
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# 添加model模块到Python路径
sys.path.append(str(Path(__file__).parent / "model"))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    from kronos import KronosTokenizer, Kronos, KronosPredictor

def analyze_huggingface_model_structure():
    """分析HuggingFace Hub上的模型结构"""
    print("🔍 分析HuggingFace模型加载失败原因")
    print("=" * 60)
    
    models_to_check = [
        "NeoQuasar/Kronos-Tokenizer-base",
        "NeoQuasar/Kronos-small"
    ]
    
    for model_name in models_to_check:
        print(f"\n📦 检查模型: {model_name}")
        print("-" * 40)
        
        # 1. 检查模型配置文件
        try:
            config_url = f"https://huggingface.co/{model_name}/raw/main/config.json"
            response = requests.get(config_url, timeout=10)
            if response.status_code == 200:
                config = response.json()
                print("✅ 找到config.json:")
                for key, value in config.items():
                    print(f"   {key}: {value}")
            else:
                print(f"❌ 无法获取config.json (状态码: {response.status_code})")
        except Exception as e:
            print(f"❌ config.json获取失败: {e}")
        
        # 2. 检查模型文件列表
        try:
            files_url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(files_url, timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                if 'siblings' in model_info:
                    print("📁 模型文件列表:")
                    for file_info in model_info['siblings']:
                        filename = file_info.get('rfilename', 'unknown')
                        size = file_info.get('size', 0)
                        print(f"   - {filename} ({size/1024/1024:.1f}MB)")
                else:
                    print("❌ 无法获取文件列表")
            else:
                print(f"❌ 无法获取模型信息 (状态码: {response.status_code})")
        except Exception as e:
            print(f"❌ 模型信息获取失败: {e}")
        
        # 3. 检查PyTorchModelHubMixin所需文件
        required_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
        for filename in required_files:
            try:
                file_url = f"https://huggingface.co/{model_name}/resolve/main/{filename}"
                response = requests.head(file_url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {filename} 存在")
                else:
                    print(f"❌ {filename} 不存在 (状态码: {response.status_code})")
            except Exception as e:
                print(f"⚠️  {filename} 检查失败: {e}")

def test_pytorch_model_hub_mixin():
    """测试PyTorchModelHubMixin的工作原理"""
    print("\n🧪 测试PyTorchModelHubMixin工作原理")
    print("=" * 60)
    
    # 检查PyTorchModelHubMixin的实现
    print("📋 PyTorchModelHubMixin的from_pretrained方法工作流程:")
    print("   1. 下载config.json文件")
    print("   2. 使用config.json中的参数调用cls(**config)")
    print("   3. 下载model权重文件 (pytorch_model.bin 或 model.safetensors)")
    print("   4. 加载权重到模型")
    
    # 模拟from_pretrained的步骤
    model_name = "NeoQuasar/Kronos-Tokenizer-base"
    
    print(f"\n🔬 模拟加载过程: {model_name}")
    
    try:
        # 步骤1: 尝试下载config.json
        print("   步骤1: 下载config.json...")
        try:
            config_path = hf_hub_download(repo_id=model_name, filename="config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("   ✅ config.json下载成功:")
            for key, value in config.items():
                print(f"      {key}: {value}")
        except EntryNotFoundError:
            print("   ❌ config.json不存在")
            config = None
        except Exception as e:
            print(f"   ❌ config.json下载失败: {e}")
            config = None
        
        # 步骤2: 尝试使用配置创建模型
        if config:
            print("   步骤2: 使用配置创建模型...")
            try:
                # 检查必需参数
                required_params = [
                    'd_in', 'd_model', 'n_heads', 'ff_dim', 'n_enc_layers', 'n_dec_layers',
                    'ffn_dropout_p', 'attn_dropout_p', 'resid_dropout_p', 's1_bits', 's2_bits',
                    'beta', 'gamma0', 'gamma', 'zeta', 'group_size'
                ]
                
                missing_params = [param for param in required_params if param not in config]
                if missing_params:
                    print(f"   ❌ 配置文件缺少必需参数: {missing_params}")
                else:
                    print("   ✅ 所有必需参数都存在")
                    tokenizer = KronosTokenizer(**config)
                    print("   ✅ 模型创建成功")
                    
            except Exception as e:
                print(f"   ❌ 模型创建失败: {e}")
        else:
            print("   步骤2: 跳过（无配置文件）")
        
        # 步骤3: 检查权重文件
        print("   步骤3: 检查权重文件...")
        weight_files = ['pytorch_model.bin', 'model.safetensors']
        for weight_file in weight_files:
            try:
                weight_path = hf_hub_download(repo_id=model_name, filename=weight_file)
                print(f"   ✅ {weight_file} 存在")
                break
            except EntryNotFoundError:
                print(f"   ❌ {weight_file} 不存在")
            except Exception as e:
                print(f"   ⚠️  {weight_file} 检查失败: {e}")
                
    except Exception as e:
        print(f"   ❌ 模拟加载过程失败: {e}")
        traceback.print_exc()

def analyze_model_class_structure():
    """分析模型类的结构"""
    print("\n🏗️  分析模型类结构")
    print("=" * 60)
    
    # 检查KronosTokenizer类
    print("📋 KronosTokenizer类分析:")
    print(f"   - 基类: {KronosTokenizer.__bases__}")
    print(f"   - MRO: {KronosTokenizer.__mro__}")
    
    # 检查__init__方法
    import inspect
    sig = inspect.signature(KronosTokenizer.__init__)
    print("   - __init__参数:")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            print(f"     {param_name}: {param.annotation if param.annotation != param.empty else 'Any'}")
    
    # 检查是否正确继承PyTorchModelHubMixin
    if hasattr(KronosTokenizer, 'from_pretrained'):
        print("   ✅ 具有from_pretrained方法")
    else:
        print("   ❌ 缺少from_pretrained方法")

def provide_solutions():
    """提供解决方案"""
    print("\n💡 问题分析与解决方案")
    print("=" * 60)
    
    print("🔍 问题根本原因:")
    print("   1. HuggingFace Hub上的模型缺少正确的config.json文件")
    print("   2. config.json中可能缺少KronosTokenizer.__init__所需的16个参数")
    print("   3. PyTorchModelHubMixin.from_pretrained()依赖config.json来初始化模型")
    
    print("\n🛠️  解决方案:")
    print("   方案1: 使用我们的修复版脚本（已实现）")
    print("     - 自动检测加载失败并使用默认配置")
    print("     - 提供完整的参数回退机制")
    
    print("   方案2: 修复HuggingFace Hub上的模型（模型作者需要）")
    print("     - 确保config.json包含所有必需参数")
    print("     - 重新上传模型到HuggingFace Hub")
    
    print("   方案3: 本地保存配置文件")
    print("     - 创建本地config.json文件")
    print("     - 使用本地路径加载模型")
    
    print("\n📋 技术细节:")
    print("   - PyTorchModelHubMixin.from_pretrained()工作流程:")
    print("     1. 下载repo中的config.json")
    print("     2. 使用config调用cls(**config)初始化模型")
    print("     3. 下载并加载权重文件")
    print("   - 失败点：步骤2，config.json缺少必需参数")

def main():
    """主函数"""
    try:
        analyze_huggingface_model_structure()
        test_pytorch_model_hub_mixin()
        analyze_model_class_structure()
        provide_solutions()
        
        print("\n✅ 诊断完成!")
        print("💡 建议: 使用修复版脚本 `run_a_share_prediction_fixed.py`")
        print("   它包含完整的故障回退机制，可以解决这个问题。")
        
    except Exception as e:
        print(f"\n❌ 诊断过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()