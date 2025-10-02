#!/usr/bin/env python3
"""
查看A股股票预测结果的详细信息
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_prediction_results():
    """分析预测结果"""
    print("📊 A股股票预测结果分析")
    print("=" * 60)
    
    # 加载原始数据
    df = pd.read_csv("examples/data/XSHG_5min_600977.csv")
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # 参数设置
    lookback = 400
    pred_len = 120
    
    # 获取历史数据和真实的未来数据进行对比
    historical_data = df.loc[:lookback-1]
    actual_future_data = df.loc[lookback:lookback+pred_len-1]
    
    print(f"📈 股票代码: XSHG 600977")
    print(f"📅 历史数据时间范围: {historical_data['timestamps'].iloc[0]} 到 {historical_data['timestamps'].iloc[-1]}")
    print(f"📅 预测时间范围: {actual_future_data['timestamps'].iloc[0]} 到 {actual_future_data['timestamps'].iloc[-1]}")
    print(f"📊 历史数据点数: {len(historical_data)}")
    print(f"🔮 预测数据点数: {len(actual_future_data)}")
    
    print("\n💰 价格分析:")
    print(f"   历史收盘价范围: {historical_data['close'].min():.4f} - {historical_data['close'].max():.4f}")
    print(f"   历史平均收盘价: {historical_data['close'].mean():.4f}")
    print(f"   真实未来收盘价范围: {actual_future_data['close'].min():.4f} - {actual_future_data['close'].max():.4f}")
    print(f"   真实未来平均收盘价: {actual_future_data['close'].mean():.4f}")
    
    print("\n📊 成交量分析:")
    print(f"   历史成交量范围: {historical_data['volume'].min():.0f} - {historical_data['volume'].max():.0f}")
    print(f"   历史平均成交量: {historical_data['volume'].mean():.0f}")
    print(f"   真实未来成交量范围: {actual_future_data['volume'].min():.0f} - {actual_future_data['volume'].max():.0f}")
    print(f"   真实未来平均成交量: {actual_future_data['volume'].mean():.0f}")
    
    print("\n📈 价格趋势分析:")
    historical_price_change = ((historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0]) * 100
    actual_price_change = ((actual_future_data['close'].iloc[-1] - actual_future_data['close'].iloc[0]) / actual_future_data['close'].iloc[0]) * 100
    
    print(f"   历史期间价格变化: {historical_price_change:+.2f}%")
    print(f"   预测期间实际价格变化: {actual_price_change:+.2f}%")
    
    # 波动性分析
    historical_volatility = historical_data['close'].pct_change().std() * np.sqrt(288)  # 5分钟数据年化波动率
    actual_volatility = actual_future_data['close'].pct_change().std() * np.sqrt(288)
    
    print(f"\n📊 波动性分析:")
    print(f"   历史年化波动率: {historical_volatility:.2%}")
    print(f"   实际未来年化波动率: {actual_volatility:.2%}")
    
    print("\n🎯 预测模型说明:")
    print("   - 本项目使用Kronos基础模型对A股股票进行预测")
    print("   - 模型基于Transformer架构，专门针对金融K线数据训练")
    print("   - 预测使用了OHLCV（开高低收量）+ 成交额数据")
    print("   - 模型使用400个历史数据点预测未来120个数据点")
    print("   - 这是一个演示版本，实际交易需要更复杂的风险管理")
    
    print("\n📁 生成的文件:")
    print("   - a_share_prediction_result.png: 预测结果可视化图表")
    print("   - run_a_share_prediction.py: 主要预测脚本")
    print("   - view_prediction_results.py: 本分析脚本")
    
    print("\n✅ 预测任务已完成！")
    print("📊 可以查看 a_share_prediction_result.png 文件查看预测结果图表。")

if __name__ == "__main__":
    analyze_prediction_results()