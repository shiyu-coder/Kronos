#!/usr/bin/env python3
"""
A股股票预测运行脚本
使用Kronos模型对A股股票进行预测
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

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
    ax1.set_title('A股股票价格预测对比', fontsize=16)

    ax2.plot(volume_df['真实成交量'], label='真实成交量', color='blue', linewidth=1.5)
    ax2.plot(volume_df['预测成交量'], label='预测成交量', color='red', linewidth=1.5)
    ax2.set_ylabel('成交量', fontsize=14)
    ax2.set_xlabel('时间', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('a_share_prediction_result.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数：运行A股股票预测"""
    print("🚀 开始运行Kronos A股股票预测模型")
    print("=" * 60)
    
    try:
        # 1. 加载模型和分词器
        print("📥 正在从Hugging Face加载预训练模型...")
        print("   - 加载分词器: NeoQuasar/Kronos-Tokenizer-base")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        
        print("   - 加载预测模型: NeoQuasar/Kronos-small")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        
        # 2. 初始化预测器
        print("🔧 初始化预测器...")
        # 使用CPU作为备选，如果GPU不可用
        try:
            predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
            print("   - 使用GPU (cuda:0) 进行预测")
        except:
            predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
            print("   - 使用CPU进行预测")
        
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
        
        # 设置预测参数
        lookback = 400  # 历史回看窗口
        pred_len = 120  # 预测长度
        
        print(f"   - 历史窗口: {lookback} 个数据点")
        print(f"   - 预测长度: {pred_len} 个数据点")
        
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
        
        print("💾 预测图表已保存为: a_share_prediction_result.png")
        print("\n🎉 A股股票预测完成!")
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()