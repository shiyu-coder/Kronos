# run_backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class KronosBacktester:
    """
    Kronos模型回测类
    """

    def __init__(self, data_dir, model_dir, initial_capital=100000):
        """
        初始化回测器

        参数:
        data_dir: 数据目录
        model_dir: 模型预测结果目录
        initial_capital: 初始资金
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        self.results = {}

    def load_historical_data(self, stock_code):
        """
        加载历史数据
        """
        csv_file = os.path.join(self.data_dir, f"{stock_code}_stock_data.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"数据文件不存在: {csv_file}")

        df = pd.read_csv(csv_file, encoding='utf-8-sig')

        # 检查列名并标准化
        column_mapping = {
            '日期': 'date',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume',
            '成交额': 'amount'
        }

        # 重命名列
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()

        print(f"✅ 加载历史数据: {len(df)} 条记录")
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")

        return df

    def load_predictions(self, stock_code):
        """
        加载模型预测结果
        """
        # 尝试不同的预测文件命名
        pred_files = [
            os.path.join(self.model_dir, f"{stock_code}_kronos_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_detailed_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_predictions.csv")
        ]

        pred_df = None
        for pred_file in pred_files:
            if os.path.exists(pred_file):
                pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')
                print(f"✅ 找到预测文件: {pred_file}")
                break

        if pred_df is None:
            raise FileNotFoundError(f"未找到预测文件，请检查目录: {self.model_dir}")

        # 标准化列名
        column_mapping = {
            '日期': 'date',
            '预测收盘价': 'predicted_close',
            '收盘价': 'predicted_close',
            '预测成交量': 'predicted_volume',
            '成交量': 'predicted_volume'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in pred_df.columns:
                pred_df = pred_df.rename(columns={old_col: new_col})

        pred_df['date'] = pd.to_datetime(pred_df['date'])
        pred_df.set_index('date', inplace=True)
        pred_df = pred_df.sort_index()

        print(f"✅ 加载预测数据: {len(pred_df)} 条记录")
        print(f"预测时间范围: {pred_df.index.min()} 到 {pred_df.index.max()}")

        return pred_df

    def align_data(self, hist_df, pred_df):
        """
        对齐历史数据和预测数据的时间范围
        """
        # 找到历史数据的最后日期
        last_hist_date = hist_df.index.max()

        # 筛选预测数据，从历史数据结束后开始
        pred_df_aligned = pred_df[pred_df.index > last_hist_date]

        if len(pred_df_aligned) == 0:
            # 如果没有未来的预测数据，使用所有预测数据
            pred_df_aligned = pred_df.copy()
            print("⚠️ 警告：预测数据没有未来的日期，使用所有预测数据")

        print(f"✅ 数据对齐: 历史数据结束于 {last_hist_date}, 预测数据从 {pred_df_aligned.index.min()} 开始")

        return pred_df_aligned

    def calculate_trading_signals(self, hist_df, pred_df, threshold=0.02):
        """
        计算交易信号
        """
        # 对齐数据
        pred_df = self.align_data(hist_df, pred_df)

        # 合并历史数据和预测数据
        combined = pd.concat([
            hist_df[['close']].rename(columns={'close': 'actual'}),
            pred_df[['predicted_close']].rename(columns={'predicted_close': 'predicted'})
        ], axis=1)

        # 计算预测收益率
        combined['pred_return'] = combined['predicted'].pct_change()

        # 生成交易信号
        combined['signal'] = 0
        combined['signal'] = np.where(combined['pred_return'] > threshold, 1,  # 买入信号
                                      np.where(combined['pred_return'] < -threshold, -1, 0))  # 卖出信号

        # 过滤信号：避免频繁交易
        combined['position'] = combined['signal'].replace(to_replace=0, method='ffill').fillna(0)

        return combined

    def run_backtest(self, combined_df):
        """
        运行回测
        """
        # 初始化资金和持仓
        capital = self.initial_capital
        position = 0
        trades = []

        # 回测记录
        backtest_results = pd.DataFrame(index=combined_df.index)
        backtest_results['capital'] = capital
        backtest_results['position'] = 0
        backtest_results['returns'] = 0.0
        backtest_results['price'] = combined_df['actual'].combine_first(combined_df['predicted'])

        for i, (date, row) in enumerate(combined_df.iterrows()):
            current_price = row['actual'] if not pd.isna(row['actual']) else row['predicted']
            signal = row['position']

            # 跳过无效价格
            if pd.isna(current_price):
                continue

            # 执行交易
            if i > 0:  # 从第二天开始
                prev_position = backtest_results['position'].iloc[i - 1] if i > 0 else 0

                # 平仓信号
                if prev_position != 0 and signal == 0:
                    # 平仓
                    capital = position * current_price
                    position = 0
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': prev_position,
                        'capital': capital
                    })

                # 开仓信号
                elif prev_position == 0 and signal != 0:
                    # 计算可买股数（假设全仓交易）
                    shares = int(capital / current_price)
                    if shares > 0:
                        position = shares * signal
                        capital -= shares * current_price
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares * signal,
                            'capital': capital
                        })

            # 更新持仓市值
            portfolio_value = capital + position * current_price

            # 记录结果
            backtest_results.loc[date, 'capital'] = portfolio_value
            backtest_results.loc[date, 'position'] = position
            backtest_results.loc[date, 'price'] = current_price

            # 计算日收益率
            if i > 0:
                prev_value = backtest_results['capital'].iloc[i - 1]
                if prev_value > 0:
                    backtest_results.loc[date, 'returns'] = (portfolio_value - prev_value) / prev_value

        return backtest_results, trades

    def calculate_metrics(self, backtest_results, trades):
        """
        计算回测指标
        """
        returns = backtest_results['returns'].replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) == 0:
            return {
                '总收益率': 0,
                '年化收益率': 0,
                '波动率': 0,
                '夏普比率': 0,
                '最大回撤': 0,
                '胜率': 0,
                '平均交易收益': 0,
                '交易次数': 0,
                '最终资金': self.initial_capital
            }

        total_return = (backtest_results['capital'].iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 交易统计
        trade_returns = []
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy = buy_trades[i]
            sell = sell_trades[i]
            trade_return = (sell['price'] - buy['price']) / buy['price']
            trade_returns.append(trade_return)

        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0

        metrics = {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '波动率': volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '胜率': win_rate,
            '平均交易收益': avg_trade_return,
            '交易次数': len(trades),
            '最终资金': backtest_results['capital'].iloc[-1]
        }

        return metrics

    def plot_backtest_results(self, backtest_results, metrics, stock_code, output_dir):
        """
        绘制回测结果图表
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # 1. 资金曲线
        ax1.plot(backtest_results.index, backtest_results['capital'],
                 linewidth=2, label='策略资金曲线', color='#1f77b4')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--',
                    label=f'初始资金 ({self.initial_capital:,.0f}元)')
        ax1.set_ylabel('资金 (元)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{stock_code} Kronos模型回测结果', fontsize=14, fontweight='bold')

        # 2. 收益率曲线
        cumulative_returns = (1 + backtest_results['returns'].fillna(0)).cumprod()
        ax2.plot(backtest_results.index, cumulative_returns,
                 linewidth=2, label='策略累计收益', color='#2ca02c')

        # 基准收益（买入持有）
        price_returns = backtest_results['price'].pct_change().fillna(0)
        benchmark_returns = (1 + price_returns).cumprod()
        ax2.plot(backtest_results.index, benchmark_returns,
                 linewidth=2, label='基准收益（买入持有）', color='#ff7f0e', alpha=0.7)

        ax2.set_ylabel('累计收益', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 回撤曲线
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        ax3.fill_between(backtest_results.index, drawdown, 0,
                         alpha=0.3, color='red', label='回撤')
        ax3.set_ylabel('回撤', fontsize=12)
        ax3.set_xlabel('日期', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 添加指标文本
        metrics_text = (
            f"总收益率: {metrics['总收益率']:.2%}\n"
            f"年化收益率: {metrics['年化收益率']:.2%}\n"
            f"夏普比率: {metrics['夏普比率']:.2f}\n"
            f"最大回撤: {metrics['最大回撤']:.2%}\n"
            f"胜率: {metrics['胜率']:.2%}\n"
            f"交易次数: {metrics['交易次数']}\n"
            f"最终资金: {metrics['最终资金']:,.0f}元"
        )

        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                    facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()

        # 保存图表
        os.makedirs(output_dir, exist_ok=True)
        chart_file = os.path.join(output_dir, f'{stock_code}_backtest_results.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 回测图表已保存: {chart_file}")

        plt.show()

    def run_complete_backtest(self, stock_code, output_dir, threshold=0.02):
        """
        运行完整的回测流程
        """
        print(f"🎯 开始 {stock_code} 回测分析")
        print("=" * 50)

        try:
            # 1. 加载数据
            print("步骤1: 加载历史数据和预测数据...")
            hist_df = self.load_historical_data(stock_code)
            pred_df = self.load_predictions(stock_code)

            # 2. 计算交易信号
            print("步骤2: 计算交易信号...")
            combined_df = self.calculate_trading_signals(hist_df, pred_df, threshold)

            # 3. 运行回测
            print("步骤3: 运行回测...")
            backtest_results, trades = self.run_backtest(combined_df)

            # 4. 计算指标
            print("步骤4: 计算回测指标...")
            metrics = self.calculate_metrics(backtest_results, trades)

            # 5. 绘制结果
            print("步骤5: 生成回测图表...")
            self.plot_backtest_results(backtest_results, metrics, stock_code, output_dir)

            # 6. 打印详细报告
            print("\n" + "=" * 70)
            print(f"📊 {stock_code} 回测报告")
            print("=" * 70)
            for key, value in metrics.items():
                if isinstance(value, float):
                    if '率' in key or '收益' in key or '回撤' in key:
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

            print(f"\n交易记录 (共{len(trades)}次交易):")
            for i, trade in enumerate(trades[-10:], 1):  # 显示最后10次交易
                print(f"  交易{i}: {trade['date'].strftime('%Y-%m-%d')} "
                      f"{trade['action']} {abs(trade['shares'])}股 @ {trade['price']:.2f}元")

            return metrics, backtest_results, trades

        except Exception as e:
            print(f"❌ 回测过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def main():
    """
    主函数：运行Kronos模型回测
    """
    # 配置参数
    BACKTEST_CONFIG = {
        "stock_code": "000831",  # 要回测的股票代码
        "data_dir": r"D:\lianghuajiaoyi\Kronos\examples\data",  # 历史数据目录
        "model_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce",  # 模型预测结果目录
        "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\backtest",  # 回测结果输出目录
        "initial_capital": 100000,  # 初始资金
        "threshold": 0.02  # 交易阈值（2%）
    }

    print("🤖 Kronos模型回测系统")
    print("=" * 50)
    print(f"回测股票: {BACKTEST_CONFIG['stock_code']}")
    print(f"初始资金: {BACKTEST_CONFIG['initial_capital']:,.0f}元")
    print(f"交易阈值: {BACKTEST_CONFIG['threshold']:.1%}")
    print()

    # 创建回测器并运行
    backtester = KronosBacktester(
        data_dir=BACKTEST_CONFIG["data_dir"],
        model_dir=BACKTEST_CONFIG["model_dir"],
        initial_capital=BACKTEST_CONFIG["initial_capital"]
    )

    metrics, results, trades = backtester.run_complete_backtest(
        stock_code=BACKTEST_CONFIG["stock_code"],
        output_dir=BACKTEST_CONFIG["output_dir"],
        threshold=BACKTEST_CONFIG["threshold"]
    )

    if metrics:
        print(f"\n✅ {BACKTEST_CONFIG['stock_code']} 回测完成!")
        print(f"📁 结果保存在: {BACKTEST_CONFIG['output_dir']}")


if __name__ == "__main__":
    main()