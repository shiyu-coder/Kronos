import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from model import Kronos, KronosTokenizer, KronosPredictor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def ensure_output_directory(output_dir):
    """确保输出目录存在，如果不存在则创建"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    return output_dir


def prepare_stock_data(csv_file_path, stock_code):
    """
    准备股票数据，转换为Kronos模型需要的格式

    参数:
    csv_file_path: CSV文件路径
    stock_code: 股票代码，用于显示信息

    返回:
    df: 处理后的DataFrame
    """
    print(f"正在加载和预处理股票 {stock_code} 数据...")

    # 读取CSV文件
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

    # 检查数据列名并重命名为标准格式
    column_mapping = {
        '日期': 'timestamps',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '成交额': 'amount'
    }

    # 只重命名存在的列
    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    # 确保时间戳列存在并转换为datetime格式
    if 'timestamps' not in df.columns:
        # 如果数据有日期索引，重置索引
        if df.index.name == '日期':
            df = df.reset_index()
            df = df.rename(columns={'日期': 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # 按时间排序
    df = df.sort_values('timestamps').reset_index(drop=True)

    print(f"✅ 数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
    print(f"数据列: {df.columns.tolist()}")

    return df


def calculate_prediction_parameters(df, target_days=100):
    """
    根据目标预测天数计算合适的参数

    参数:
    df: 股票数据DataFrame
    target_days: 目标预测天数（自然日）

    返回:
    lookback: 回看期数
    pred_len: 预测期数
    """
    # 计算平均交易日数量（考虑节假日）
    total_days = (df['timestamps'].max() - df['timestamps'].min()).days
    trading_days = len(df)
    trading_ratio = trading_days / total_days if total_days > 0 else 0.7  # 交易日比例

    # 计算目标预测的交易日数量
    pred_trading_days = int(target_days * trading_ratio)

    # 设置回看期数为预测期数的2-3倍，但不超过数据总量的70%
    max_lookback = int(len(df) * 0.7)
    lookback = min(pred_trading_days * 2, max_lookback, len(df) - pred_trading_days)
    pred_len = min(pred_trading_days, len(df) - lookback)

    print(f"📊 参数计算:")
    print(f"  目标预测天数: {target_days} 天（自然日）")
    print(f"  预计交易日数量: {pred_trading_days} 天")
    print(f"  回看期数 (lookback): {lookback}")
    print(f"  预测期数 (pred_len): {pred_len}")

    return lookback, pred_len


def generate_future_dates_with_holidays(last_date, pred_len):
    """
    生成未来的交易日日期，考虑中国节假日

    参数:
    last_date: 最后一个历史数据的日期
    pred_len: 预测期数

    返回:
    future_dates: 未来的交易日日期列表
    """
    # 中国主要节假日（需要根据实际情况调整）
    holidays_2025 = [
        # 2025年国庆节假期（通常为10月1日-10月8日）
        datetime(2025, 10, 1), datetime(2025, 10, 2), datetime(2025, 10, 3),
        datetime(2025, 10, 4), datetime(2025, 10, 5), datetime(2025, 10, 6),
        datetime(2025, 10, 7), datetime(2025, 10, 8),  # 添加10月8日
        # 周末调休等可以根据需要添加
    ]

    future_dates = []
    current_date = last_date + timedelta(days=1)

    while len(future_dates) < pred_len:
        # 如果是工作日（周一到周五）且不是节假日
        if current_date.weekday() < 5 and current_date not in holidays_2025:
            future_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"📅 生成的未来交易日: 共 {len(future_dates)} 天")
    print(f"   起始日期: {future_dates[0].strftime('%Y-%m-%d')}")
    print(f"   结束日期: {future_dates[-1].strftime('%Y-%m-%d')}")

    # 显示节假日信息
    holiday_count = sum(1 for date in holidays_2025 if date > last_date)
    print(f"   包含节假日: {holiday_count} 天")

    return future_dates[:pred_len]


def plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code="002354", stock_name="股票", pred_len=100,
                                 output_dir="."):
    """
    绘制详细的预测结果图表 - 优化版，图表更大更清晰

    参数:
    kline_df: 历史K线数据
    pred_df: 预测数据
    future_dates: 未来日期列表
    stock_code: 股票代码
    stock_name: 股票名称
    pred_len: 预测期数
    output_dir: 输出目录
    """
    # 确保输出目录存在
    ensure_output_directory(output_dir)

    # 确保数据长度一致
    min_len = min(len(pred_df), len(future_dates))
    pred_df = pred_df.iloc[:min_len]
    future_dates = future_dates[:min_len]

    # 设置预测数据的索引为未来日期
    pred_df.index = future_dates

    # 准备价格数据
    sr_close = kline_df.set_index('timestamps')['close']
    sr_pred_close = pred_df['close']
    sr_close.name = '历史数据'
    sr_pred_close.name = "预测数据"

    # 准备成交量数据
    sr_volume = kline_df.set_index('timestamps')['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = '历史数据'
    sr_pred_volume.name = "预测数据"

    # 合并数据
    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    # 创建更大的图表
    fig = plt.figure(figsize=(18, 14))

    # 使用GridSpec创建更灵活的布局
    gs = plt.GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1])

    ax1 = fig.add_subplot(gs[0])  # 价格图表
    ax2 = fig.add_subplot(gs[1])  # 成交量图表
    ax3 = fig.add_subplot(gs[2])  # 价格变动图表

    # 1. 价格图表 - 更大更清晰
    # 只显示最近200个交易日的历史数据，避免图表过于拥挤
    recent_history = close_df['历史数据'].iloc[-min(200, len(close_df['历史数据'])):]
    ax1.plot(recent_history.index, recent_history.values, label='历史价格', color='#1f77b4', linewidth=2.5, alpha=0.9)
    ax1.plot(close_df['预测数据'].index, close_df['预测数据'].values, label='预测价格',
             color='#ff7f0e', linewidth=2.5, linestyle='-', marker='o', markersize=3)

    # 添加预测起始点的标记
    prediction_start_date = close_df['预测数据'].index[0] if len(close_df['预测数据']) > 0 else close_df.index[-1]
    prediction_start_price = close_df['历史数据'].iloc[-1]
    ax1.axvline(x=prediction_start_date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.annotate('预测起点', xy=(prediction_start_date, prediction_start_price),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_ylabel('收盘价 (元)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{stock_name}({stock_code}) 股票价格预测 - 未来{pred_len}个交易日',
                  fontsize=16, fontweight='bold', pad=20)

    # 设置x轴日期格式
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 设置y轴格式
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    # 2. 成交量图表 - 优化显示
    # 只显示预测期的成交量
    pred_volumes = volume_df['预测数据'].dropna()
    if len(pred_volumes) > 0:
        ax2.bar(pred_volumes.index, pred_volumes.values,
                alpha=0.7, color='#ff7f0e', label='预测成交量', width=0.8)

    ax2.set_ylabel('成交量 (手)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 设置x轴标签
    if len(pred_volumes) > 0:
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. 价格变动图表 - 优化显示
    if len(close_df['预测数据']) > 0:
        price_change = close_df['预测数据'] - close_df['历史数据'].iloc[-1]
        colors = ['green' if x >= 0 else 'red' for x in price_change]

        # 每5个交易日显示一个标签，避免过于拥挤
        bars = ax3.bar(range(len(price_change)), price_change, alpha=0.8, color=colors)

        # 在关键点添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if i % 10 == 0 or i == len(bars) - 1 or abs(height) > price_change.std():  # 每10天或最后一天或显著波动
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:+.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=8, fontweight='bold')

        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    ax3.set_ylabel('价格变动 (元)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('交易日', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 设置x轴标签
    if len(price_change) > 0:
        # 每10个交易日显示一个标签
        xticks_positions = list(range(0, len(price_change), max(1, len(price_change) // 10)))
        if len(price_change) - 1 not in xticks_positions:
            xticks_positions.append(len(price_change) - 1)
        ax3.set_xticks(xticks_positions)
        ax3.set_xticklabels([f'D{i + 1}' for i in xticks_positions])

    # 添加详细的统计信息框
    if len(close_df['预测数据']) > 0 and not np.isnan(close_df['历史数据'].iloc[-1]):
        pred_stats = {
            '股票代码': stock_code,
            '股票名称': stock_name,
            '当前价格': f"{close_df['历史数据'].iloc[-1]:.2f} 元",
            '预测结束价格': f"{close_df['预测数据'].iloc[-1]:.2f} 元",
            '预测涨跌幅': f"{(close_df['预测数据'].iloc[-1] / close_df['历史数据'].iloc[-1] - 1) * 100:+.2f}%",
            '预测期间最高价': f"{close_df['预测数据'].max():.2f} 元",
            '预测期间最低价': f"{close_df['预测数据'].min():.2f} 元",
            '预测波动率': f"{close_df['预测数据'].std():.2f} 元",
            '预测起始日期': f"{close_df['预测数据'].index[0].strftime('%Y-%m-%d')}",
            '预测结束日期': f"{close_df['预测数据'].index[-1].strftime('%Y-%m-%d')}",
            '预测交易日数': f"{len(close_df['预测数据'])} 天"
        }

        stats_text = "\n".join([f"{k}: {v}" for k, v in pred_stats.items()])
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                 verticalalignment='bottom')

    plt.tight_layout()

    # 保存高分辨率图片到指定目录
    chart_filename = os.path.join(output_dir, f'{stock_code}_prediction_chart.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 预测图表已保存: {chart_filename}")

    plt.show()

    return close_df, volume_df


def generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code="002354", stock_name="股票",
                               output_dir="."):
    """
    生成预测报告
    """
    # 确保输出目录存在
    ensure_output_directory(output_dir)

    print(f"\n{'=' * 70}")
    print(f"📊 {stock_name}({stock_code}) 股票预测报告")
    print(f"{'=' * 70}")

    if len(close_df['预测数据']) == 0 or np.isnan(close_df['历史数据'].iloc[-1]):
        print("❌ 没有有效的预测数据可生成报告")
        return

    # 确保所有数组长度一致
    min_len = min(len(close_df['预测数据']), len(volume_df['预测数据']), len(future_dates))

    # 基本统计
    historical_close = close_df['历史数据'].iloc[-1]
    predicted_close = close_df['预测数据'].iloc[-1]
    price_change_pct = (predicted_close / historical_close - 1) * 100

    print(f"🔮 预测概览:")
    print(f"   当前价格: {historical_close:.2f} 元")
    print(f"   预测结束价格: {predicted_close:.2f} 元")
    print(f"   预测涨跌幅: {price_change_pct:+.2f}%")
    print(f"   预测期间: {min_len} 个交易日")
    print(
        f"   预测时间范围: {future_dates[0].strftime('%Y-%m-%d')} 到 {future_dates[min_len - 1].strftime('%Y-%m-%d')}")

    print(f"\n📈 价格预测统计:")
    print(f"   预测最高价: {close_df['预测数据'].max():.2f} 元")
    print(f"   预测最低价: {close_df['预测数据'].min():.2f} 元")
    print(f"   预测平均价: {close_df['预测数据'].mean():.2f} 元")
    print(f"   价格波动率: {close_df['预测数据'].std():.2f} 元")

    print(f"\n📊 成交量预测统计:")
    print(f"   预测平均成交量: {volume_df['预测数据'].mean():,.0f} 手")
    print(f"   预测最大成交量: {volume_df['预测数据'].max():,.0f} 手")
    print(f"   预测最小成交量: {volume_df['预测数据'].min():,.0f} 手")

    # 保存详细预测数据到指定目录 - 确保所有数组长度一致
    prediction_details = pd.DataFrame({
        '日期': future_dates[:min_len],
        '预测收盘价': close_df['预测数据'].values[:min_len],
        '预测成交量': volume_df['预测数据'].values[:min_len],
        '价格变动(元)': (close_df['预测数据'].values[:min_len] - historical_close),
        '价格变动(%)': ((close_df['预测数据'].values[:min_len] / historical_close - 1) * 100)
    })

    prediction_file = os.path.join(output_dir, f'{stock_code}_detailed_predictions.csv')
    prediction_details.to_csv(prediction_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 详细预测数据已保存: {prediction_file}")


def main(stock_code="002354", stock_name="天娱数科", data_dir="./data", pred_days=100, output_dir="./output"):
    """
    主函数：执行股票价格预测

    参数:
    stock_code: 股票代码
    stock_name: 股票名称
    data_dir: 数据文件目录
    pred_days: 预测天数（自然日）
    output_dir: 输出文件目录
    """
    # 构建数据文件路径
    csv_file_path = os.path.join(data_dir, f"{stock_code}_stock_data.csv")

    print(f"🎯 开始 {stock_name}({stock_code}) 股票价格预测")
    print("=" * 70)
    print(f"数据文件: {csv_file_path}")
    print(f"预测天数: {pred_days} 天（自然日）")
    print(f"输出目录: {output_dir}")

    # 检查数据文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"❌ 数据文件不存在: {csv_file_path}")
        print("请先运行数据获取脚本生成股票数据文件")
        return

    # 确保输出目录存在
    ensure_output_directory(output_dir)

    try:
        # 1. 加载模型和分词器
        print("\n步骤1: 加载Kronos模型和分词器...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        print("✅ 模型加载完成")

        # 2. 实例化预测器
        print("步骤2: 初始化预测器...")
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
        print("✅ 预测器初始化完成")

        # 3. 准备数据
        print("步骤3: 准备股票数据...")
        df = prepare_stock_data(csv_file_path, stock_code)

        # 4. 计算预测参数
        print("步骤4: 计算预测参数...")
        lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)

        if pred_len <= 0:
            print("❌ 数据量不足，无法进行预测")
            return

        print(f"✅ 最终参数 - 回看期: {lookback}, 预测期: {pred_len}")

        # 5. 准备输入数据
        print("步骤5: 准备输入数据...")
        # 使用最新的数据作为输入
        x_df = df.loc[-lookback:, ['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.loc[-lookback:, 'timestamps'].reset_index(drop=True)

        # 生成未来日期（考虑节假日）
        last_historical_date = df['timestamps'].iloc[-1]
        future_dates = generate_future_dates_with_holidays(last_historical_date, pred_len)

        print(f"输入数据形状: {x_df.shape}")
        print(f"历史数据时间范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
        print(f"预测时间范围: {future_dates[0]} 到 {future_dates[-1]}")

        # 6. 执行预测
        print("步骤6: 执行价格预测...")
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(future_dates),  # 使用未来日期作为预测时间戳
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )

        print("✅ 预测完成")

        # 7. 显示预测结果
        print("\n步骤7: 显示预测结果...")
        print("预测数据前5行:")
        # 确保预测数据长度与未来日期一致
        min_len = min(len(pred_df), len(future_dates))
        pred_df = pred_df.iloc[:min_len]
        pred_df.index = future_dates[:min_len]
        print(pred_df.head())

        # 8. 可视化结果
        print("步骤8: 生成可视化图表...")
        # 使用最后一部分历史数据和预测数据
        kline_df = df.loc[-lookback:].reset_index(drop=True)
        close_df, volume_df = plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code, stock_name,
                                                           pred_len, output_dir)

        # 9. 生成预测报告
        print("步骤9: 生成预测报告...")
        generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code, stock_name, output_dir)

        print(f"\n🎉 {stock_name}({stock_code}) 股票预测完成!")
        print("生成的文件:")
        print(f"  📊 {os.path.join(output_dir, stock_code + '_prediction_chart.png')} - 预测图表")
        print(f"  📋 {os.path.join(output_dir, stock_code + '_detailed_predictions.csv')} - 详细预测数据")

        # 显示预测总结
        if len(close_df['预测数据']) > 0 and not np.isnan(close_df['历史数据'].iloc[-1]):
            print(f"\n📈 预测总结:")
            historical_price = close_df['历史数据'].iloc[-1]
            predicted_price = close_df['预测数据'].iloc[-1]
            change_pct = (predicted_price / historical_price - 1) * 100

            print(f"  当前价格: {historical_price:.2f} 元")
            print(f"  预测价格: {predicted_price:.2f} 元")
            print(f"  预期涨跌: {change_pct:+.2f}%")
            print(
                f"  预测时间: {future_dates[0].strftime('%Y-%m-%d')} 到 {future_dates[min_len - 1].strftime('%Y-%m-%d')}")

            if change_pct > 10:
                print(f"  🚀 模型预测未来{pred_len}个交易日大幅看涨 (+{change_pct:.1f}%)")
            elif change_pct > 5:
                print(f"  📈 模型预测未来{pred_len}个交易日看涨 (+{change_pct:.1f}%)")
            elif change_pct > 0:
                print(f"  ↗️ 模型预测未来{pred_len}个交易日微涨 (+{change_pct:.1f}%)")
            elif change_pct > -5:
                print(f"  ↘️ 模型预测未来{pred_len}个交易日微跌 ({change_pct:.1f}%)")
            elif change_pct > -10:
                print(f"  📉 模型预测未来{pred_len}个交易日看跌 ({change_pct:.1f}%)")
            else:
                print(f"  🔻 模型预测未来{pred_len}个交易日大幅看跌 ({change_pct:.1f}%)")

    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# 使用方法说明
if __name__ == "__main__":
    """
    股票预测工具 - 支持多股票预测

    使用方法：
    修改下面的 STOCK_CONFIG 来预测不同的股票
    """

    # ==================== 在这里修改股票配置 ====================
    STOCK_CONFIG = {
        "stock_code": "300418",  # 股票代码
        "stock_name": "昆仑万维",  # 股票名称
        "data_dir": "./data",  # 数据文件目录
        "pred_days": 100,  # 预测100个自然日
        "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce"  # 输出文件目录
    }

    # 其他股票配置示例：
    # STOCK_CONFIG = {"stock_code": "000001", "stock_name": "平安银行", "data_dir": "./data", "pred_days": 100, "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce"}
    # STOCK_CONFIG = {"stock_code": "600036", "stock_name": "招商银行", "data_dir": "./data", "pred_days": 100, "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce"}
    # STOCK_CONFIG = {"stock_code": "300750", "stock_name": "宁德时代", "data_dir": "./data", "pred_days": 100, "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce"}
    # =========================================================

    print("🤖 智能股票预测工具")
    print("=" * 70)
    print(f"当前预测股票: {STOCK_CONFIG['stock_name']}({STOCK_CONFIG['stock_code']})")
    print(f"数据目录: {STOCK_CONFIG['data_dir']}")
    print(f"预测天数: {STOCK_CONFIG['pred_days']} 天（自然日）")
    print(f"输出目录: {STOCK_CONFIG['output_dir']}")
    print()

    # 运行主程序
    main(**STOCK_CONFIG)

    print(f"\n💡 提示：要预测其他股票，请修改代码中的 STOCK_CONFIG 变量")