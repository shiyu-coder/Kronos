#!/usr/bin/env python3
"""
A股股票预测运行脚本 - 修复版本
使用Kronos模型对A股股票进行预测
修复了模型加载配置问题
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore")

# 添加model模块到Python路径
sys.path.append(str(Path(__file__).parent / "model"))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    # 如果无法导入，尝试直接从kronos模块导入
    from kronos import KronosTokenizer, Kronos, KronosPredictor

def plot_prediction(kline_df, pred_df):
    """绘制预测结果对比图"""
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = '真实价格'
    sr_pred_close.name = "预测价格"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = '真实成交量'
    sr_pred_volume.name = "预测成交量"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(close_df['真实价格'], label='真实价格', color='blue', linewidth=1.5)
    ax1.plot(close_df['预测价格'], label='预测价格', color='red', linewidth=1.5)
    ax1.set_ylabel('收盘价', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)
    ax1.set_title('A股股票价格预测对比 (修复版)', fontsize=16)

    ax2.plot(volume_df['真实成交量'], label='真实成交量', color='blue', linewidth=1.5)
    ax2.plot(volume_df['预测成交量'], label='预测成交量', color='red', linewidth=1.5)
    ax2.set_ylabel('成交量', fontsize=14)
    ax2.set_xlabel('时间', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('a_share_prediction_result_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_tokenizer_with_config():
    """创建具有默认配置的tokenizer"""
    # 根据Kronos-Tokenizer-base的典型配置
    tokenizer_config = {
        'd_in': 6,  # OHLCV + amount (6 features)
        'd_model': 256,
        'n_heads': 8,
        'ff_dim': 1024,
        'n_enc_layers': 4,
        'n_dec_layers': 4,
        'ffn_dropout_p': 0.1,
        'attn_dropout_p': 0.1,
        'resid_dropout_p': 0.1,
        's1_bits': 8,
        's2_bits': 8,
        'beta': 0.25,
        'gamma0': 0.99,
        'gamma': 0.99,
        'zeta': 1.0,
        'group_size': 4
    }
    
    print("   - 使用默认配置创建tokenizer...")
    for key, value in tokenizer_config.items():
        print(f"     {key}: {value}")
    
    return KronosTokenizer(**tokenizer_config)

def create_model_with_config():
    """创建具有默认配置的模型"""
    # 根据Kronos-small的典型配置
    model_config = {
        's1_bits': 8,
        's2_bits': 8,
        'n_layers': 12,
        'd_model': 256,
        'n_heads': 8,
        'ff_dim': 1024,
        'ffn_dropout_p': 0.1,
        'attn_dropout_p': 0.1,
        'resid_dropout_p': 0.1,
        'token_dropout_p': 0.1,
        'learn_te': True
    }
    
    print("   - 使用默认配置创建模型...")
    for key, value in model_config.items():
        print(f"     {key}: {value}")
    
    return Kronos(**model_config)

def load_models_with_fallback():
    """尝试从HuggingFace加载模型，失败时使用默认配置"""
    tokenizer = None
    model = None
    
    # 尝试从HuggingFace加载
    try:
        print("📥 尝试从Hugging Face加载预训练模型...")
        print("   - 加载分词器: NeoQuasar/Kronos-Tokenizer-base")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        print("   ✅ 分词器加载成功")
        
        print("   - 加载预测模型: NeoQuasar/Kronos-small")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("   ✅ 模型加载成功")
        
    except Exception as e:
        print(f"   ❌ HuggingFace加载失败: {str(e)}")
        print("   🔧 使用默认配置创建模型...")
        
        # 使用默认配置创建模型
        try:
            tokenizer = create_tokenizer_with_config()
            model = create_model_with_config()
            print("   ✅ 默认配置模型创建成功")
            print("   ⚠️  注意：使用默认配置，预测结果可能不如预训练模型准确")
        except Exception as e2:
            print(f"   ❌ 默认配置创建也失败: {str(e2)}")
            raise RuntimeError("无法创建模型，请检查模型配置")
    
    return tokenizer, model

def main():
    """主函数：运行A股股票预测"""
    print("🚀 开始运行Kronos A股股票预测模型 (修复版)")
    print("=" * 60)
    
    try:
        # 1. 加载模型和分词器（带故障回退）
        tokenizer, model = load_models_with_fallback()
        
        # 2. 初始化预测器
        print("🔧 初始化预测器...")
        # 使用CPU作为备选，如果GPU不可用
        try:
            # 检测是否有CUDA可用
            if torch.cuda.is_available():
                device = "cuda:0"
                print("   - 检测到CUDA，使用GPU进行预测")
            else:
                device = "cpu"
                print("   - 未检测到CUDA，使用CPU进行预测")
                
            predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
            print(f"   - 预测器初始化成功 (设备: {device})")
        except Exception as e:
            device = "cpu"
            print(f"   - GPU初始化失败，降级到CPU: {str(e)}")
            predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        
        # 3. 准备数据
        print("📊 准备A股数据...")
        data_path = "examples/data/XSHG_5min_600977.csv"
        
        if not os.path.exists(data_path):
            print(f"❌ 数据文件未找到: {data_path}")
            print("请确保数据文件存在!")
            return
            
        df = pd.read_csv(data_path)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        print(f"   - 数据文件: {data_path}")
        print(f"   - 数据范围: {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")
        print(f"   - 总数据点: {len(df)}")
        
        # 设置预测参数 (减少计算量以适配默认配置)
        lookback = min(400, len(df) - 121)  # 确保有足够数据
        pred_len = 60  # 减少预测长度以降低计算复杂度
        
        print(f"   - 历史窗口: {lookback} 个数据点")
        print(f"   - 预测长度: {pred_len} 个数据点")
        
        # 检查数据是否足够
        if len(df) < lookback + pred_len:
            print(f"❌ 数据不足：需要至少 {lookback + pred_len} 个数据点，实际有 {len(df)} 个")
            return
        
        # 准备输入数据
        x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df.loc[:lookback-1, 'timestamps']
        y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
        
        # 4. 执行预测
        print("🔮 正在进行A股股票预测...")
        print("   这可能需要几分钟时间，请耐心等待...")
        
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,          # 温度参数
            top_p=0.9,      # 核采样概率
            sample_count=1, # 预测路径数量
            verbose=True
        )
        
        # 5. 展示结果
        print("\n✅ 预测完成!")
        print("=" * 60)
        print("📈 预测结果预览:")
        print(pred_df.head())
        
        print("\n📊 预测统计信息:")
        print(f"   - 预测数据点数: {len(pred_df)}")
        print(f"   - 预测价格范围: {pred_df['close'].min():.4f} - {pred_df['close'].max():.4f}")
        print(f"   - 预测成交量范围: {pred_df['volume'].min():.0f} - {pred_df['volume'].max():.0f}")
        
        # 6. 生成可视化
        print("\n📊 生成预测结果图表...")
        kline_df = df.loc[:lookback+pred_len-1]
        plot_prediction(kline_df, pred_df)
        
        print("💾 预测图表已保存为: a_share_prediction_result_fixed.png")
        print("\n🎉 A股股票预测完成!")
        
        # 额外信息
        print("\n💡 提示:")
        print("   - 如果使用了默认配置，预测精度可能不如预训练模型")
        print("   - 建议在有稳定网络连接时重新运行以下载预训练权重")
        print("   - 可以根据实际需求调整预测参数(温度、top_p等)")
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()