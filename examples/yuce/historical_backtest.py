# historical_backtest.py
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


class HistoricalBacktester:
    """
    历史回测类：用历史数据验证模型预测效果
    """

    def __init__(self, data_dir, initial_capital=100000):
        self.data_dir = data_dir
        self.initial_capital = initial_capital

    def load_historical_data(self, stock_code):
        """加载历史数据"""
        csv_file = os.path.join(self.data_dir, f"{stock_code}_stock_data.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"数据文件不存在: {csv_file}")

        df = pd.read_csv(csv_file, encoding='utf-8-sig')

        # 标准化列名
        column_mapping = {
            '日期': 'date',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume',
            '成交额': 'amount'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()

        print(f"✅ 加载历史数据: {len(df)} 条记录")
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")

        return df

    def simulate_model_prediction(self, df, lookback_days=60, pred_days=30):
        """
        模拟模型预测：使用历史数据进行"预测"，然后与实际结果对比
        """
        results = []

        # 从数据中选取多个时间点进行"预测"
        test_points = range(lookback_days, len(df) - pred_days, pred_days)

        for start_idx in test_points:
            # 模拟预测：使用前lookback_days天数据"预测"后pred_days天
            historical_data = df.iloc[start_idx - lookback_days:start_idx]
            actual_future = df.iloc[start_idx:start_idx + pred_days]

            # 简单的预测策略（这里应该替换为您的实际模型预测）
            # 这里使用移动平均作为示例预测
            pred_close = self.simple_prediction(historical_data, pred_days)

            # 记录结果
            for i in range(min(len(pred_close), len(actual_future))):
                results.append({
                    'date': actual_future.index[i],
                    'actual_close': actual_future['close'].iloc[i],
                    'predicted_close': pred_close[i],
                    'lookback_start': historical_data.index[0],
                    'prediction_date': historical_data.index[-1]
                })

        return pd.DataFrame(results)

    def simple_prediction(self, historical_data, pred_days):
        """简单的预测方法（示例）"""
        # 使用移动平均 + 随机波动作为预测
        last_price = historical_data['close'].iloc[-1]
        avg_volatility = historical_data['close'].pct_change().std()

        predictions = []
        current_price = last_price

        for _ in range(pred_days):
            # 模拟价格变化（正态分布）
            change = np.random.normal(0, avg_volatility)
            current_price = current_price * (1 + change)
            predictions.append(current_price)

        return predictions

    def calculate_prediction_accuracy(self, results_df):
        """计算预测准确率"""
        results_df['error'] = results_df['predicted_close'] - results_df['actual_close']
        results_df['error_pct'] = results_df['error'] / results_df['actual_close']
        results_df['abs_error_pct'] = abs(results_df['error_pct'])

        accuracy_metrics = {
            '平均绝对误差率': results_df['abs_error_pct'].mean(),
            '预测准确率': (results_df['abs_error_pct'] < 0.05).mean(),  # 误差小于5%算准确
            '方向准确率': (np.sign(results_df['predicted_close'].diff()) ==
                           np.sign(results_df['actual_close'].diff())).mean(),
            '相关系数': results_df['predicted_close'].corr(results_df['actual_close'])
        }

        return accuracy_metrics

    def run_trading_strategy(self, results_df, threshold=0.03):
        """基于预测结果运行交易策略"""
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = []

        # 按日期排序
        results_df = results_df.sort_index()

        for date, row in results_df.iterrows():
            current_price = row['actual_close']
            predicted_price = row['predicted_close']
            predicted_return = (predicted_price - current_price) / current_price

            # 交易逻辑
            if position == 0 and predicted_return > threshold:
                # 买入信号
                shares = int(capital / current_price)
                if shares > 0:
                    position = shares
                    capital -= shares * current_price
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'reason': f'预测上涨{predicted_return:.2%}'
                    })

            elif position > 0 and predicted_return < -threshold:
                # 卖出信号
                capital += position * current_price
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'reason': f'预测下跌{predicted_return:.2%}'
                })
                position = 0

            # 计算当前资产总值
            portfolio_value = capital + position * current_price
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'position': position,
                'price': current_price
            })

        return pd.DataFrame(portfolio_values), trades

    def calculate_performance(self, portfolio_df, trades):
        """计算策略表现"""
        portfolio_df = portfolio_df.set_index('date')
        returns = portfolio_df['portfolio_value'].pct_change().dropna()

        total_return = (portfolio_df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital

        if len(returns) > 0:
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.03) / volatility if volatility > 0 else 0

            # 最大回撤
            cumulative = (1 + returns).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
        else:
            annual_return = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0

        # 买入持有策略对比
        buy_hold_return = (portfolio_df['price'].iloc[-1] - portfolio_df['price'].iloc[0]) / portfolio_df['price'].iloc[
            0]

        performance = {
            '策略总收益': total_return,
            '策略年化收益': annual_return,
            '买入持有收益': buy_hold_return,
            '波动率': volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '交易次数': len(trades),
            '最终资金': portfolio_df['portfolio_value'].iloc[-1],
            '超额收益': total_return - buy_hold_return
        }

        return performance

    def plot_comparison(self, results_df, portfolio_df, stock_code, output_dir):
        """绘制预测对比图表"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # 1. 价格预测对比
        ax1.plot(results_df.index, results_df['actual_close'],
                 label='实际价格', color='blue', linewidth=2)
        ax1.plot(results_df.index, results_df['predicted_close'],
                 label='预测价格', color='red', linestyle='--', alpha=0.7)
        ax1.set_ylabel('价格 (元)')
        ax1.legend()
        ax1.set_title(f'{stock_code} - 价格预测 vs 实际走势', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. 预测误差
        ax2.bar(results_df.index, results_df['error_pct'] * 100,
                alpha=0.6, color='orange')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('预测误差 (%)')
        ax2.set_title('预测误差分析')
        ax2.grid(True, alpha=0.3)

        # 3. 策略表现
        ax3.plot(portfolio_df['date'], portfolio_df['portfolio_value'],
                 label='策略资金曲线', color='green', linewidth=2)
        ax3.axhline(y=self.initial_capital, color='red', linestyle='--',
                    label=f'初始资金 ({self.initial_capital:,.0f}元)')

        # 买入持有对比
        initial_shares = self.initial_capital / portfolio_df['price'].iloc[0]
        buy_hold_values = portfolio_df['price'] * initial_shares
        ax3.plot(portfolio_df['date'], buy_hold_values,
                 label='买入持有策略', color='blue', linestyle=':', alpha=0.7)

        ax3.set_ylabel('资金 (元)')
        ax3.set_xlabel('日期')
        ax3.legend()
        ax3.set_title('策略表现对比')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        os.makedirs(output_dir, exist_ok=True)
        chart_file = os.path.join(output_dir, f'{stock_code}_historical_backtest.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 历史回测图表已保存: {chart_file}")

        plt.show()

    def run_complete_backtest(self, stock_code, output_dir, lookback_days=60, pred_days=30, threshold=0.03):
        """运行完整的历史回测"""
        print(f"🎯 开始 {stock_code} 历史回测分析")
        print("=" * 60)

        try:
            # 1. 加载历史数据
            print("步骤1: 加载历史数据...")
            df = self.load_historical_data(stock_code)

            # 2. 模拟模型预测
            print("步骤2: 模拟模型预测...")
            results_df = self.simulate_model_prediction(df, lookback_days, pred_days)

            # 3. 计算预测准确率
            print("步骤3: 计算预测准确率...")
            accuracy_metrics = self.calculate_prediction_accuracy(results_df)

            # 4. 运行交易策略
            print("步骤4: 运行交易策略...")
            portfolio_df, trades = self.run_trading_strategy(results_df, threshold)

            # 5. 计算策略表现
            print("步骤5: 计算策略表现...")
            performance = self.calculate_performance(portfolio_df, trades)

            # 6. 绘制结果
            print("步骤6: 生成回测图表...")
            self.plot_comparison(results_df, portfolio_df, stock_code, output_dir)

            # 7. 打印报告
            print("\n" + "=" * 70)
            print(f"📊 {stock_code} 历史回测报告")
            print("=" * 70)

            print("\n🔍 预测准确率分析:")
            for metric, value in accuracy_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value:.4f}")

            print("\n💰 策略表现分析:")
            for metric, value in performance.items():
                if isinstance(value, float):
                    if '收益' in metric or '回撤' in metric:
                        print(f"  {metric}: {value:.2%}")
                    else:
                        print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

            print(f"\n📈 交易统计:")
            print(f"  总交易次数: {len(trades)}")
            print(f"  买入次数: {len([t for t in trades if t['action'] == 'BUY'])}")
            print(f"  卖出次数: {len([t for t in trades if t['action'] == 'SELL'])}")

            if len(trades) > 0:
                print(f"\n最近5次交易:")
                for trade in trades[-5:]:
                    print(f"  {trade['date'].strftime('%Y-%m-%d')} {trade['action']} "
                          f"{trade['shares']}股 @ {trade['price']:.2f}元 - {trade['reason']}")

            return accuracy_metrics, performance, results_df

        except Exception as e:
            print(f"❌ 回测过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def main():
    """主函数"""
    # 配置参数
    BACKTEST_CONFIG = {
        "stock_code": "300418",
        "data_dir": r"D:\lianghuajiaoyi\Kronos\examples\data",
        "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\historical_backtest",
        "initial_capital": 100000,
        "lookback_days": 60,  # 使用60天历史数据
        "pred_days": 30,  # 预测30天
        "threshold": 0.03  # 3%的交易阈值
    }

    print("🤖 Kronos模型历史回测系统")
    print("=" * 50)
    print(f"回测股票: {BACKTEST_CONFIG['stock_code']}")
    print(f"回看天数: {BACKTEST_CONFIG['lookback_days']}天")
    print(f"预测天数: {BACKTEST_CONFIG['pred_days']}天")
    print(f"初始资金: {BACKTEST_CONFIG['initial_capital']:,.0f}元")
    print()

    # 创建回测器并运行
    backtester = HistoricalBacktester(
        data_dir=BACKTEST_CONFIG["data_dir"],
        initial_capital=BACKTEST_CONFIG["initial_capital"]
    )

    accuracy, performance, results = backtester.run_complete_backtest(
        stock_code=BACKTEST_CONFIG["stock_code"],
        output_dir=BACKTEST_CONFIG["output_dir"],
        lookback_days=BACKTEST_CONFIG["lookback_days"],
        pred_days=BACKTEST_CONFIG["pred_days"],
        threshold=BACKTEST_CONFIG["threshold"]
    )

    if accuracy and performance:
        print(f"\n✅ {BACKTEST_CONFIG['stock_code']} 历史回测完成!")

        # 简单结论
        if performance['超额收益'] > 0:
            print("🎉 结论: 模型策略跑赢了买入持有策略!")
        else:
            print("⚠️ 结论: 模型策略未能跑赢买入持有策略。")

        print(f"📁 详细结果保存在: {BACKTEST_CONFIG['output_dir']}")


if __name__ == "__main__":
    main()