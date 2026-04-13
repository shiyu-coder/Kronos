import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
import requests
import json
import time
import random
import akshare as ak
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    print("⚠️ 无法导入Kronos模型，预测功能将不可用")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class StockPredictorGUI:
    """股票预测图形界面"""

    def __init__(self, root):
        self.root = root
        self.root.title("Kronos股票预测系统")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        # 初始化市场分析器
        self.market_analyzer = EnhancedMarketFactorAnalyzer()

        # 创建界面
        self.create_widgets()

        # 默认配置
        self.default_config = {
            "stock_code": "600580",
            "stock_name": "卧龙电驱",
            "data_dir": r"D:\lianghuajiaoyi\Kronos\examples\data",
            "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce",
            "pred_days": 60,
            "history_years": 1
        }

    def create_widgets(self):
        """创建界面组件"""
        # 主标题
        title_label = tk.Label(
            self.root,
            text="🤖 Kronos股票预测系统",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=10)

        # 说明标签
        desc_label = tk.Label(
            self.root,
            text="基于Kronos模型的多维度股票价格预测系统",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        desc_label.pack(pady=5)

        # 创建主框架
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 输入框架
        input_frame = tk.LabelFrame(main_frame, text="股票参数设置", font=("Arial", 11, "bold"),
                                    bg='#f0f0f0', fg='#2c3e50')
        input_frame.pack(fill=tk.X, pady=10)

        # 股票代码输入
        tk.Label(input_frame, text="股票代码:", bg='#f0f0f0', font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W,
                                                                                       padx=5, pady=5)
        self.stock_code_var = tk.StringVar(value="600580")
        stock_code_entry = tk.Entry(input_frame, textvariable=self.stock_code_var, font=("Arial", 10), width=15)
        stock_code_entry.grid(row=0, column=1, padx=5, pady=5)

        # 股票名称输入
        tk.Label(input_frame, text="股票名称:", bg='#f0f0f0', font=("Arial", 10)).grid(row=0, column=2, sticky=tk.W,
                                                                                       padx=5, pady=5)
        self.stock_name_var = tk.StringVar(value="卧龙电驱")
        stock_name_entry = tk.Entry(input_frame, textvariable=self.stock_name_var, font=("Arial", 10), width=15)
        stock_name_entry.grid(row=0, column=3, padx=5, pady=5)

        # 预测天数
        tk.Label(input_frame, text="预测天数:", bg='#f0f0f0', font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W,
                                                                                       padx=5, pady=5)
        self.pred_days_var = tk.StringVar(value="60")
        pred_days_entry = tk.Entry(input_frame, textvariable=self.pred_days_var, font=("Arial", 10), width=15)
        pred_days_entry.grid(row=1, column=1, padx=5, pady=5)

        # 历史数据年限
        tk.Label(input_frame, text="历史年限:", bg='#f0f0f0', font=("Arial", 10)).grid(row=1, column=2, sticky=tk.W,
                                                                                       padx=5, pady=5)
        self.history_years_var = tk.StringVar(value="1")
        history_years_entry = tk.Entry(input_frame, textvariable=self.history_years_var, font=("Arial", 10), width=15)
        history_years_entry.grid(row=1, column=3, padx=5, pady=5)

        # 目录设置框架
        dir_frame = tk.LabelFrame(main_frame, text="目录设置", font=("Arial", 11, "bold"),
                                  bg='#f0f0f0', fg='#2c3e50')
        dir_frame.pack(fill=tk.X, pady=10)

        # 数据目录
        tk.Label(dir_frame, text="数据目录:", bg='#f0f0f0', font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W,
                                                                                     padx=5, pady=5)
        self.data_dir_var = tk.StringVar(value=r"D:\lianghuajiaoyi\Kronos\examples\data")
        data_dir_entry = tk.Entry(dir_frame, textvariable=self.data_dir_var, font=("Arial", 10), width=40)
        data_dir_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(dir_frame, text="浏览", command=self.browse_data_dir, font=("Arial", 9)).grid(row=0, column=2, padx=5,
                                                                                                pady=5)

        # 输出目录
        tk.Label(dir_frame, text="输出目录:", bg='#f0f0f0', font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W,
                                                                                     padx=5, pady=5)
        self.output_dir_var = tk.StringVar(value=r"D:\lianghuajiaoyi\Kronos\examples\yuce")
        output_dir_entry = tk.Entry(dir_frame, textvariable=self.output_dir_var, font=("Arial", 10), width=40)
        output_dir_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(dir_frame, text="浏览", command=self.browse_output_dir, font=("Arial", 9)).grid(row=1, column=2,
                                                                                                  padx=5, pady=5)

        # 功能按钮框架
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=20)

        # 预测按钮
        self.predict_button = tk.Button(
            button_frame,
            text="🚀 开始预测",
            command=self.start_prediction,
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            width=15,
            height=2
        )
        self.predict_button.pack(side=tk.LEFT, padx=10)

        # 重置按钮
        reset_button = tk.Button(
            button_frame,
            text="🔄 重置",
            command=self.reset_fields,
            font=("Arial", 10),
            bg='#95a5a6',
            fg='white',
            width=10,
            height=2
        )
        reset_button.pack(side=tk.LEFT, padx=10)

        # 退出按钮
        exit_button = tk.Button(
            button_frame,
            text="❌ 退出",
            command=self.root.quit,
            font=("Arial", 10),
            bg='#e74c3c',
            fg='white',
            width=10,
            height=2
        )
        exit_button.pack(side=tk.LEFT, padx=10)

        # 进度显示
        self.progress_frame = tk.LabelFrame(main_frame, text="预测进度", font=("Arial", 11, "bold"),
                                            bg='#f0f0f0', fg='#2c3e50')
        self.progress_frame.pack(fill=tk.X, pady=10)

        self.progress_var = tk.StringVar(value="等待开始预测...")
        progress_label = tk.Label(self.progress_frame, textvariable=self.progress_var, bg='#f0f0f0',
                                  font=("Arial", 10), wraplength=700, justify=tk.LEFT)
        progress_label.pack(padx=10, pady=10, fill=tk.X)

        # 进度条
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # 结果展示区域
        self.result_frame = tk.LabelFrame(main_frame, text="预测结果", font=("Arial", 11, "bold"),
                                          bg='#f0f0f0', fg='#2c3e50')
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.result_text = tk.Text(self.result_frame, height=8, font=("Arial", 9), wrap=tk.WORD)
        scrollbar = tk.Scrollbar(self.result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def browse_data_dir(self):
        """浏览数据目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.data_dir_var.set(directory)

    def browse_output_dir(self):
        """浏览输出目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def reset_fields(self):
        """重置输入字段"""
        self.stock_code_var.set("600580")
        self.stock_name_var.set("卧龙电驱")
        self.pred_days_var.set("60")
        self.history_years_var.set("1")
        self.data_dir_var.set(r"D:\lianghuajiaoyi\Kronos\examples\data")
        self.output_dir_var.set(r"D:\lianghuajiaoyi\Kronos\examples\yuce")
        self.result_text.delete(1.0, tk.END)
        self.progress_var.set("等待开始预测...")

    def start_prediction(self):
        """开始预测"""
        # 验证输入
        if not self.validate_inputs():
            return

        # 禁用预测按钮
        self.predict_button.config(state=tk.DISABLED)

        # 清空结果区域
        self.result_text.delete(1.0, tk.END)

        # 开始进度条
        self.progress_bar.start()

        # 在新线程中运行预测
        prediction_thread = threading.Thread(target=self.run_prediction)
        prediction_thread.daemon = True
        prediction_thread.start()

    def validate_inputs(self):
        """验证输入参数"""
        try:
            stock_code = self.stock_code_var.get().strip()
            stock_name = self.stock_name_var.get().strip()
            pred_days = int(self.pred_days_var.get())
            history_years = int(self.history_years_var.get())

            if not stock_code:
                messagebox.showerror("错误", "请输入股票代码")
                return False

            if not stock_name:
                messagebox.showerror("错误", "请输入股票名称")
                return False

            if pred_days <= 0 or pred_days > 365:
                messagebox.showerror("错误", "预测天数应在1-365天之间")
                return False

            if history_years <= 0 or history_years > 10:
                messagebox.showerror("错误", "历史年限应在1-10年之间")
                return False

            return True

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
            return False

    def run_prediction(self):
        """运行预测流程"""
        try:
            # 获取输入参数
            stock_code = self.stock_code_var.get().strip()
            stock_name = self.stock_name_var.get().strip()
            pred_days = int(self.pred_days_var.get())
            history_years = int(self.history_years_var.get())
            data_dir = self.data_dir_var.get()
            output_dir = self.output_dir_var.get()

            # 更新进度
            self.update_progress("🎯 开始股票预测流程...")

            # 运行预测
            success, result = run_comprehensive_prediction_gui(
                stock_code, stock_name, data_dir, pred_days, output_dir, history_years,
                progress_callback=self.update_progress,
                result_callback=self.update_result
            )

            if success:
                self.update_progress("✅ 预测完成！")
                messagebox.showinfo("完成", f"{stock_name}({stock_code})预测完成！\n图表已保存到输出目录。")
            else:
                self.update_progress("❌ 预测失败")
                messagebox.showerror("错误", f"预测失败: {result}")

        except Exception as e:
            self.update_progress(f"❌ 预测过程出现错误: {str(e)}")
            messagebox.showerror("错误", f"预测过程出现错误: {str(e)}")
        finally:
            # 重新启用预测按钮
            self.root.after(0, lambda: self.predict_button.config(state=tk.NORMAL))
            # 停止进度条
            self.root.after(0, self.progress_bar.stop)

    def update_progress(self, message):
        """更新进度信息"""
        self.root.after(0, lambda: self.progress_var.set(message))
        print(message)  # 同时在控制台输出

    def update_result(self, message):
        """更新结果信息"""
        self.root.after(0, lambda: self.result_text.insert(tk.END, message + "\n"))
        self.root.after(0, lambda: self.result_text.see(tk.END))


# ==================== 基础数据获取函数 ====================
def ensure_output_directory(output_dir):
    """确保输出目录存在，如果不存在则创建"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    return output_dir


def fetch_real_stock_data(stock_code, period="daily", adjust="qfq"):
    """
    使用AKShare获取真实股票数据
    """
    try:
        print(f"📡 正在通过AKShare获取 {stock_code} 的真实股票数据...")

        # 获取股票数据
        df = ak.stock_zh_a_hist(symbol=stock_code, period=period, adjust=adjust)

        if df is None or df.empty:
            print(f"❌ 未获取到 {stock_code} 的数据")
            return None

        # 重命名列以统一格式
        column_mapping = {
            '日期': 'timestamps',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change_amount',
            '换手率': 'turnover'
        }

        # 只映射存在的列
        actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=actual_mapping)

        # 确保时间戳格式正确
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)

        # 添加股票代码列
        df['stock_code'] = stock_code

        print(f"✅ 成功获取 {len(df)} 条真实数据")
        print(f"📈 最新收盘价: {df['close'].iloc[-1]:.2f}元, 涨跌幅: {df['pct_chg'].iloc[-1]:.2f}%")
        print(f"📅 时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")

        return df

    except Exception as e:
        print(f"❌ AKShare数据获取失败: {e}")
        return None


def get_stock_data_with_retry_all_history(stock_code="600580", retry_count=2):
    """
    优化的数据获取函数 - 优先使用真实API数据
    """
    print(f"🔄 尝试获取股票 {stock_code} 的真实历史数据...")

    # 优先使用AKShare获取真实数据
    df = fetch_real_stock_data(stock_code, "daily", "qfq")

    if df is not None:
        return df
    else:
        print("⚠️ 真实数据获取失败，使用基于真实价格的模拟数据...")
        return create_realistic_fallback_data(stock_code)


def create_realistic_fallback_data(stock_code="600580"):
    """
    基于真实价格的备用数据生成函数
    """
    # 基于真实市场价格的参考数据
    real_stock_references = {
        '600580': {'name': '卧龙电驱', 'current_price': 15.20, 'range': (12.0, 20.0)},
        '300207': {'name': '欣旺达', 'current_price': 33.79, 'range': (28.0, 38.0)},
        '300418': {'name': '昆仑万维', 'current_price': 48.59, 'range': (40.0, 55.0)},
        '002354': {'name': '天娱数科', 'current_price': 15.20, 'range': (12.0, 20.0)},
        '000001': {'name': '平安银行', 'current_price': 12.50, 'range': (10.0, 16.0)},
        '600036': {'name': '招商银行', 'current_price': 35.80, 'range': (30.0, 42.0)},
    }

    stock_info = real_stock_references.get(stock_code, {
        'name': '未知股票',
        'current_price': 20.0,
        'range': (15.0, 25.0)
    })

    # 生成最近1年的交易日数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.bdate_range(start=start_date, end=end_date, freq='B')

    # 生成基于真实价格的价格序列
    np.random.seed(42)
    n_points = len(dates)

    # 从当前价格反向生成历史价格
    current_price = stock_info['current_price']
    min_price, max_price = stock_info['range']

    # 反向生成价格序列
    prices = [current_price]
    for i in range(1, n_points):
        volatility = 0.02
        historical_return = np.random.normal(-0.0002, volatility)

        prev_price = prices[0] * (1 + historical_return)
        prev_price = max(min_price * 0.9, min(max_price * 1.1, prev_price))
        prices.insert(0, prev_price)

    # 生成OHLC数据
    stock_data = []
    for i, date in enumerate(dates):
        close_price = prices[i]

        daily_volatility = abs(np.random.normal(0, 0.015))
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + daily_volatility)
        low_price = min(open_price, close_price) * (1 - daily_volatility)

        high_price = max(open_price, close_price, low_price, high_price)
        low_price = min(open_price, close_price, high_price, low_price)

        volume = int(abs(np.random.normal(1500000, 400000)))
        amount = volume * close_price

        if i > 0:
            pct_chg = ((close_price - prices[i - 1]) / prices[i - 1]) * 100
            change_amount = close_price - prices[i - 1]
        else:
            pct_chg = 0
            change_amount = 0

        stock_data.append({
            'timestamps': date,
            'stock_code': stock_code,
            'open': round(open_price, 2),
            'close': round(close_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'volume': volume,
            'amount': round(amount, 2),
            'amplitude': round(((high_price - low_price) / open_price) * 100, 2),
            'pct_chg': round(pct_chg, 2),
            'change_amount': round(change_amount, 2),
            'turnover': round(np.random.uniform(3.0, 8.0), 2)
        })

    df = pd.DataFrame(stock_data)
    print(f"✅ 已生成基于真实价格的备用数据 {len(df)} 条")
    return df


def save_all_history_stock_data(df, stock_code, save_dir):
    """
    保存股票数据到指定目录
    """
    if df is not None and not df.empty:
        os.makedirs(save_dir, exist_ok=True)
        csv_file = os.path.join(save_dir, f"{stock_code}_stock_data.csv")
        df_reset = df.reset_index()
        df_reset.to_csv(csv_file, encoding='utf-8-sig', index=False)
        print(f"📁 股票数据已保存: {csv_file}")
        return True
    return False


def get_stock_data(stock_code, data_dir):
    """
    获取股票数据，如果数据文件不存在则从API获取真实数据
    """
    csv_file_path = os.path.join(data_dir, f"{stock_code}_stock_data.csv")

    if os.path.exists(csv_file_path):
        print(f"📁 使用现有数据文件: {csv_file_path}")
        return True, csv_file_path
    else:
        print(f"📡 数据文件不存在，从API获取真实数据...")
        df = get_stock_data_with_retry_all_history(stock_code)

        if df is not None and not df.empty:
            save_all_history_stock_data(df, stock_code, data_dir)
            return True, csv_file_path
        else:
            print(f"❌ 无法获取股票数据")
            return False, None


def prepare_stock_data(csv_file_path, stock_code, history_years=1):
    """
    准备股票数据，转换为Kronos模型需要的格式
    """
    print(f"正在加载和预处理股票 {stock_code} 数据...")

    # 读取CSV文件
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

    # 标准化列名
    column_mapping = {
        '日期': 'timestamps',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '成交额': 'amount',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low'
    }

    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    # 确保时间戳列存在并转换为datetime格式
    if 'timestamps' not in df.columns:
        if df.index.name == '日期':
            df = df.reset_index()
            df = df.rename(columns={'日期': 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    # 根据历史年限筛选数据
    if history_years > 0:
        cutoff_date = datetime.now() - timedelta(days=history_years * 365)
        original_count = len(df)
        df = df[df['timestamps'] >= cutoff_date]
        print(f"📅 使用最近 {history_years} 年数据: {len(df)} 条记录 (从 {original_count} 条中筛选)")

    # 数据验证
    print(f"🔍 数据验证 - 最近5个交易日收盘价:")
    recent_prices = df[['timestamps', 'close']].tail()
    for _, row in recent_prices.iterrows():
        print(f"  {row['timestamps'].strftime('%Y-%m-%d')}: {row['close']:.2f}元")

    current_price = df['close'].iloc[-1]
    print(f"✅ 数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
    print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"当前价格: {current_price:.2f}元")

    return df


def calculate_prediction_parameters(df, target_days=60):
    """
    根据目标预测天数计算合适的参数
    """
    # 计算平均交易日数量
    total_days = (df['timestamps'].max() - df['timestamps'].min()).days
    trading_days = len(df)
    trading_ratio = trading_days / total_days if total_days > 0 else 0.7

    # 计算目标预测的交易日数量
    pred_trading_days = int(target_days * trading_ratio)

    # 设置回看期数
    max_lookback = int(len(df) * 0.7)
    lookback = min(pred_trading_days * 3, max_lookback, len(df) - pred_trading_days)
    pred_len = min(pred_trading_days, len(df) - lookback)

    # 确保参数在合理范围内
    lookback = max(100, min(lookback, 400))
    pred_len = max(20, min(pred_len, 120))

    print(f"📊 参数计算:")
    print(f"  目标预测天数: {target_days} 天（自然日）")
    print(f"  预计交易日数量: {pred_trading_days} 天")
    print(f"  回看期数 (lookback): {lookback}")
    print(f"  预测期数 (pred_len): {pred_len}")

    return lookback, pred_len


def generate_trading_dates_only(last_date, pred_len):
    """
    🎯 修复版：只生成交易日，排除周末和法定节假日
    """
    # 2025年法定节假日安排（修正版）
    holidays_2025 = [
        '2025-01-01',  # 元旦
        '2025-01-27', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02',  # 春节
        '2025-04-04', '2025-04-05', '2025-04-06',  # 清明
        '2025-05-01', '2025-05-02', '2025-05-03',  # 劳动节
        '2025-06-08', '2025-06-09', '2025-06-10',  # 端午
        '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05', '2025-10-06', '2025-10-07',  # 国庆节
    ]

    holidays = [datetime.strptime(date, '%Y-%m-%d').date() for date in holidays_2025]

    trading_dates = []
    current_date = last_date + timedelta(days=1)

    while len(trading_dates) < pred_len:
        # 排除周末和节假日
        if current_date.weekday() < 5 and current_date.date() not in holidays:
            trading_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"📅 生成的纯交易日: 共 {len(trading_dates)} 天")
    if trading_dates:
        print(f"   起始: {trading_dates[0].strftime('%Y-%m-%d')}")
        print(f"   结束: {trading_dates[-1].strftime('%Y-%m-%d')}")

    return trading_dates


def calculate_optimal_interval(min_val, max_val):
    """
    计算最优的Y轴刻度间隔
    """
    range_val = max_val - min_val
    if range_val <= 0:
        return 1.0

    if range_val < 1:
        interval = 0.1
    elif range_val < 5:
        interval = 0.5
    elif range_val < 10:
        interval = 1.0
    elif range_val < 20:
        interval = 2.0
    elif range_val < 50:
        interval = 5.0
    elif range_val < 100:
        interval = 10.0
    elif range_val < 200:
        interval = 20.0
    elif range_val < 500:
        interval = 50.0
    else:
        interval = 100.0

    return interval


# ==================== 增强版市场因素分析器 ====================
class EnhancedMarketFactorAnalyzer:
    """增强版市场因素分析器 - 整合更多维度的市场因素"""

    def __init__(self):
        self.market_data = {}
        self.sector_data = {}
        self.macro_factors = {}
        self.policy_factors = {}

    def analyze_market_trend(self, index_codes=["000001", "399001"]):
        """
        分析大盘趋势 - 多指数综合分析
        """
        try:
            print(f"📊 综合分析大盘趋势...")

            market_analysis = {}

            for index_code in index_codes:
                index_name = "上证指数" if index_code == "000001" else "深证成指"
                print(f"  分析{index_name}({index_code})...")

                # 获取指数数据
                index_df = ak.stock_zh_index_hist(symbol=index_code, period="daily")

                if index_df is None or index_df.empty:
                    print(f"  ❌ 无法获取{index_name}数据")
                    continue

                # 重命名列
                index_df = index_df.rename(columns={
                    '日期': 'date', '收盘': 'close', '开盘': 'open',
                    '最高': 'high', '最低': 'low', '成交量': 'volume'
                })
                index_df['date'] = pd.to_datetime(index_df['date'])
                index_df = index_df.sort_values('date').reset_index(drop=True)

                # 计算技术指标
                index_df['ma5'] = index_df['close'].rolling(5).mean()
                index_df['ma20'] = index_df['close'].rolling(20).mean()
                index_df['ma60'] = index_df['close'].rolling(60).mean()
                index_df['vol_ma5'] = index_df['volume'].rolling(5).mean()

                # 技术分析
                current_data = index_df.iloc[-1]
                prev_data = index_df.iloc[-2]

                # 均线多头排列判断
                ma_condition = (current_data['ma5'] > current_data['ma20'] > current_data['ma60'])

                # 价格站在20日均线以上
                price_above_ma20 = current_data['close'] > current_data['ma20']

                # 成交量配合
                volume_condition = current_data['volume'] > current_data['vol_ma5'] * 0.8

                # 趋势强度
                trend_strength = self._calculate_trend_strength(index_df)

                is_main_uptrend = ma_condition and price_above_ma20 and trend_strength > 0.6

                market_analysis[index_name] = {
                    'is_main_uptrend': is_main_uptrend,
                    'trend_strength': trend_strength,
                    'current_close': current_data['close'],
                    'price_change_pct': ((current_data['close'] - prev_data['close']) / prev_data['close']) * 100,
                    'market_status': '主升浪' if is_main_uptrend else '震荡调整'
                }

            # 综合判断
            if market_analysis:
                avg_trend_strength = np.mean([data['trend_strength'] for data in market_analysis.values()])
                uptrend_count = sum(1 for data in market_analysis.values() if data['is_main_uptrend'])
                overall_uptrend = uptrend_count >= len(market_analysis) * 0.5

                final_analysis = {
                    'overall_is_main_uptrend': overall_uptrend,
                    'overall_trend_strength': avg_trend_strength,
                    'detailed_analysis': market_analysis,
                    'market_status': '主升浪' if overall_uptrend else '震荡调整'
                }

                print(f"✅ 大盘分析完成: {final_analysis['market_status']}, 综合趋势强度: {avg_trend_strength:.2f}")
                return final_analysis

            return self._get_default_market_analysis()

        except Exception as e:
            print(f"❌ 大盘分析错误: {e}")
            return self._get_default_market_analysis()

    def analyze_sector_resonance(self, stock_code):
        """
        分析板块共振效应 - 增强版行业分析
        """
        try:
            print(f"🔄 分析板块共振效应...")

            # 获取股票所属行业和概念
            industry = "未知"
            concepts = []

            try:
                stock_info = ak.stock_individual_info_em(symbol=stock_code)
                if not stock_info.empty and 'value' in stock_info.columns:
                    industry_row = stock_info[stock_info['item'] == '行业']
                    if not industry_row.empty:
                        industry = industry_row['value'].iloc[0]
            except:
                pass

            # 热门板块和概念映射
            hot_sectors = {
                '机器人': {'momentum': 0.85, 'limit_up_stocks': 18, 'active': True,
                           'description': '人形机器人、工业自动化'},
                '半导体': {'momentum': 0.8, 'limit_up_stocks': 15, 'active': True, 'description': '芯片国产替代'},
                '人工智能': {'momentum': 0.75, 'limit_up_stocks': 12, 'active': True, 'description': 'AI大模型、算力'},
                '低空经济': {'momentum': 0.7, 'limit_up_stocks': 10, 'active': True, 'description': '无人机、eVTOL'},
                '新能源': {'momentum': 0.6, 'limit_up_stocks': 8, 'active': True, 'description': '光伏、储能'},
                '医药': {'momentum': 0.5, 'limit_up_stocks': 5, 'active': False, 'description': '创新药'}
            }

            # 判断当前股票所属热门板块
            matched_sectors = []
            for sector, data in hot_sectors.items():
                if (sector in industry or
                        (stock_code == '600580' and sector in ['机器人', '低空经济']) or  # 卧龙电驱特殊处理
                        (stock_code == '300207' and sector in ['新能源'])):
                    matched_sectors.append({
                        'sector': sector,
                        'momentum': data['momentum'],
                        'limit_up_stocks': data['limit_up_stocks'],
                        'is_active': data['active'],
                        'description': data['description']
                    })

            # 计算综合共振分数
            if matched_sectors:
                resonance_score = np.mean([sector['momentum'] for sector in matched_sectors])
                is_sector_hot = any(sector['is_active'] for sector in matched_sectors)
                main_sector = max(matched_sectors, key=lambda x: x['momentum'])
            else:
                resonance_score = 0.5
                is_sector_hot = False
                main_sector = {'sector': '传统行业', 'momentum': 0.5, 'description': '无热门概念'}

            analysis = {
                'industry': industry,
                'matched_sectors': matched_sectors,
                'main_sector': main_sector,
                'is_sector_hot': is_sector_hot,
                'resonance_score': resonance_score,
                'sector_count': len(matched_sectors)
            }

            print(f"✅ 板块分析完成: {industry}, 匹配{len(matched_sectors)}个热门板块, 共振分数: {resonance_score:.2f}")
            return analysis

        except Exception as e:
            print(f"❌ 板块分析错误: {e}")
            return self._get_default_sector_analysis()

    def analyze_macro_factors(self):
        """
        分析宏观因素 - 结合国内外政策
        """
        try:
            print(f"🌍 分析宏观因素...")

            # 美国降息周期分析 - 基于最新信息
            us_rate_analysis = {
                'current_rate': 4.25,  # 联邦基金利率目标区间4.00%-4.25%
                'trend': '降息周期',
                'recent_cut': '2025年9月降息25个基点',
                'expected_cuts_2025': 2,  # 市场预期2025年还有两次降息
                'expected_cuts_2026': 2,
                'impact_on_emerging_markets': 'positive',
                'usd_index_support': 95.0,  # 美元指数短期支撑位
                'analysis': '美联储开启宽松周期，利好全球流动性'
            }

            # 国内政策因素 - 基于最新政策
            domestic_policy = {
                'monetary_policy': '稳健偏松',
                'fiscal_policy': '积极财政',
                'market_liquidity': '合理充裕',
                'industrial_policy': '设备更新、以旧换新',  # 大规模设备更新政策
                'employment_policy': '稳就业政策加力',  # 国务院稳就业政策
                'analysis': '政策组合拳发力，经济稳中向好'
            }

            # 行业政策支持
            industry_policy = {
                'robot_policy': '机器人产业政策支持',
                'chip_policy': '国产替代加速推进',
                'AI_policy': '人工智能发展规划',
                'low_altitude': '低空经济发展规划'
            }

            macro_analysis = {
                'us_rate_cycle': us_rate_analysis,
                'domestic_policy': domestic_policy,
                'industry_policy': industry_policy,
                'global_liquidity_outlook': '改善',
                'overall_macro_score': 0.75  # 宏观环境整体偏积极
            }

            print(
                f"✅ 宏观分析完成: 美国{us_rate_analysis['trend']}, 国内政策积极, 宏观评分: {macro_analysis['overall_macro_score']:.2f}")
            return macro_analysis

        except Exception as e:
            print(f"❌ 宏观分析错误: {e}")
            return self._get_default_macro_analysis()

    def analyze_company_fundamentals(self, stock_code):
        """
        分析公司基本面 - 针对特定股票
        """
        try:
            print(f"🏢 分析公司基本面...")

            # 卧龙电驱特殊分析
            if stock_code == '600580':
                fundamentals = {
                    'company_name': '卧龙电驱',
                    'business_areas': ['工业电机', '机器人关键部件', '航空电机', '新能源汽车驱动'],
                    'recent_developments': [
                        '与智元机器人实现双向持股，推进具身智能机器人技术研发',
                        '成立浙江龙飞电驱，专注航空电机业务',
                        '发布AI外骨骼机器人及灵巧手',
                        '布局高爆发关节模组、伺服驱动器等人形机器人关键部件'
                    ],
                    'growth_drivers': [
                        '设备更新政策推动工业电机需求',
                        '机器人产业快速发展',
                        '低空经济政策支持',
                        '出海战略加速'
                    ],
                    'risk_factors': [
                        '机器人业务营收占比仅2.71%，占比较低',
                        '工业需求景气度波动',
                        '原料价格波动风险'
                    ],
                    'investment_rating': '积极关注',
                    'fundamental_score': 0.7
                }
            else:
                # 其他股票的基础分析
                fundamentals = {
                    'company_name': '未知',
                    'business_areas': [],
                    'recent_developments': [],
                    'growth_drivers': [],
                    'risk_factors': [],
                    'investment_rating': '中性',
                    'fundamental_score': 0.5
                }

            print(f"✅ 基本面分析完成: {fundamentals['company_name']}, 评分: {fundamentals['fundamental_score']:.2f}")
            return fundamentals

        except Exception as e:
            print(f"❌ 基本面分析错误: {e}")
            return self._get_default_fundamental_analysis()

    def _calculate_trend_strength(self, df):
        """计算趋势强度"""
        if len(df) < 20:
            return 0.5

        ma_slope = (df['ma5'].iloc[-1] - df['ma5'].iloc[-20]) / df['ma5'].iloc[-20]
        price_slope = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]

        volume_trend = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean()

        strength = (ma_slope * 0.4 + price_slope * 0.4 + min(volume_trend - 1, 0.2) * 0.2)
        return max(0, min(1, strength * 10))

    def _get_default_market_analysis(self):
        return {
            'overall_is_main_uptrend': False,
            'overall_trend_strength': 0.5,
            'market_status': '未知',
            'detailed_analysis': {}
        }

    def _get_default_sector_analysis(self):
        return {
            'industry': '未知',
            'matched_sectors': [],
            'main_sector': {'sector': '未知', 'momentum': 0.5, 'description': ''},
            'is_sector_hot': False,
            'resonance_score': 0.5,
            'sector_count': 0
        }

    def _get_default_macro_analysis(self):
        return {
            'us_rate_cycle': {'trend': '未知', 'expected_cuts_2025': 0},
            'domestic_policy': {'monetary_policy': '中性'},
            'overall_macro_score': 0.5
        }

    def _get_default_fundamental_analysis(self):
        return {
            'company_name': '未知',
            'business_areas': [],
            'recent_developments': [],
            'growth_drivers': [],
            'risk_factors': [],
            'investment_rating': '中性',
            'fundamental_score': 0.5
        }


# ==================== 优化的预测平滑函数 ====================
def smooth_prediction_results(prediction_df, historical_df, smooth_factor=0.3):
    """
    🎯 优化预测结果的平滑处理，避免剧烈波动
    """
    print("🔄 应用预测结果平滑处理...")

    smoothed_df = prediction_df.copy()

    # 获取历史数据的趋势
    recent_trend = calculate_recent_trend(historical_df)

    # 对价格序列进行平滑
    price_columns = ['close', 'open', 'high', 'low']
    for col in price_columns:
        if col in smoothed_df.columns:
            original_values = smoothed_df[col].values

            # 应用移动平均平滑
            window_size = max(3, min(7, len(original_values) // 5))
            smoothed_values = pd.Series(original_values).rolling(
                window=window_size, center=True, min_periods=1
            ).mean()

            # 结合历史趋势进行微调
            trend_adjusted = smoothed_values * (1 + recent_trend * smooth_factor)

            smoothed_df[col] = trend_adjusted.values

    # 对成交量进行合理调整
    if 'volume' in smoothed_df.columns:
        hist_volume_mean = historical_df['volume'].tail(20).mean()
        current_volume = smoothed_df['volume'].values

        # 保持成交量在合理范围内
        volume_factor = 0.8 + 0.4 * np.random.random(len(current_volume))
        adjusted_volume = current_volume * volume_factor

        # 确保成交量不会异常波动
        volume_std = historical_df['volume'].tail(50).std()
        volume_min = hist_volume_mean * 0.3
        volume_max = hist_volume_mean * 3.0

        smoothed_df['volume'] = np.clip(adjusted_volume, volume_min, volume_max)

    print("✅ 预测结果平滑完成")
    return smoothed_df


def calculate_recent_trend(historical_df, lookback_days=20):
    """
    计算近期价格趋势
    """
    if len(historical_df) < lookback_days:
        lookback_days = len(historical_df)

    recent_prices = historical_df['close'].tail(lookback_days).values
    if len(recent_prices) < 2:
        return 0

    # 计算线性回归斜率作为趋势
    x = np.arange(len(recent_prices))
    slope = np.polyfit(x, recent_prices, 1)[0]

    # 归一化为趋势强度 (-1 到 1)
    price_range = np.ptp(recent_prices)
    if price_range > 0:
        trend_strength = slope / price_range * len(recent_prices)
    else:
        trend_strength = 0

    return np.clip(trend_strength, -0.1, 0.1)  # 限制趋势强度


def apply_post_holiday_adjustment(prediction_df, future_dates, holiday_periods):
    """
    🎯 修复版：应用节后调整，避免国庆后异常下跌
    """
    print("🔄 应用节后日历效应调整...")

    adjusted_df = prediction_df.copy()

    for holiday in holiday_periods:
        holiday_start = pd.Timestamp(holiday['start'])
        holiday_end = pd.Timestamp(holiday['end'])
        adjustment_days = holiday['adjustment_days']
        effect_strength = holiday['effect_strength']

        # 计算调整期结束日期
        adjustment_end = holiday_end + timedelta(days=adjustment_days)

        # 找到在节后调整期内的日期索引
        post_holiday_indices = []
        for i, date in enumerate(future_dates):
            if holiday_end <= date < adjustment_end:
                post_holiday_indices.append(i)

        # 应用节后效应调整
        if post_holiday_indices:
            for col in ['close', 'open', 'high', 'low']:
                if col in adjusted_df.columns:
                    for idx in post_holiday_indices:
                        adjusted_df.iloc[idx][col] = adjusted_df.iloc[idx][col] * (1 + effect_strength)

    print("✅ 节后调整完成")
    return adjusted_df


# ==================== 价格合理性检查函数 ====================
def validate_prediction_results(historical_df, prediction_df, max_price_change=0.3):
    """
    🎯 验证预测结果的合理性，避免异常价格波动
    """
    print("🔍 验证预测结果合理性...")

    validated_df = prediction_df.copy()
    current_price = historical_df['close'].iloc[-1]

    # 检查价格列的合理性
    price_columns = ['close', 'open', 'high', 'low']

    for col in price_columns:
        if col in validated_df.columns:
            # 计算最大允许的价格变化范围
            max_allowed_change = current_price * max_price_change

            # 检查每个预测价格
            for i in range(len(validated_df)):
                predicted_price = validated_df[col].iloc[i]

                # 如果预测价格超出合理范围，进行修正
                if abs(predicted_price - current_price) > max_allowed_change:
                    # 基于历史波动率进行修正
                    correction_factor = 0.8 + 0.4 * np.random.random()
                    corrected_price = current_price * (1 + (predicted_price / current_price - 1) * correction_factor)
                    validated_df.iloc[i][col] = corrected_price

                    print(f"⚠️  修正异常{col}价格: {predicted_price:.2f} -> {corrected_price:.2f}")

    print("✅ 预测结果验证完成")
    return validated_df


# ==================== GUI版本预测函数 ====================
def run_comprehensive_prediction_gui(stock_code, stock_name, data_dir, pred_days, output_dir, history_years=1,
                                     progress_callback=None, result_callback=None):
    """
    GUI版本的预测函数
    """

    def update_progress(message):
        if progress_callback:
            progress_callback(message)
        print(message)

    def update_result(message):
        if result_callback:
            result_callback(message)
        print(message)

    try:
        # 初始化市场分析器
        market_analyzer = EnhancedMarketFactorAnalyzer()

        update_progress(f"🎯 开始 {stock_name}({stock_code}) 预测流程")
        update_progress("=" * 50)

        # 1. 获取数据
        update_progress("\n步骤1: 获取股票数据...")
        success, csv_file_path = get_stock_data(stock_code, data_dir)
        if not success:
            update_result("❌ 无法获取股票数据，预测终止")
            return False, "无法获取股票数据"

        # 2. 加载模型和分词器
        update_progress("\n步骤2: 加载Kronos模型和分词器...")
        try:
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
            update_progress("✅ 模型加载完成 - 使用Kronos-base模型")
        except Exception as e:
            error_msg = f"❌ 模型加载失败: {e}"
            update_result(error_msg)
            update_progress("⚠️ 预测功能不可用，请检查模型安装")
            return False, error_msg

        # 3. 实例化预测器
        update_progress("步骤3: 初始化预测器...")
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
        update_progress("✅ 预测器初始化完成")

        # 4. 准备数据
        update_progress("步骤4: 准备股票数据...")
        df = prepare_stock_data(csv_file_path, stock_code, history_years)

        # 5. 计算预测参数
        update_progress("步骤5: 计算预测参数...")
        lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)

        if pred_len <= 0:
            update_result("❌ 数据量不足，无法进行预测")
            return False, "数据量不足"

        update_progress(f"✅ 最终参数 - 回看期: {lookback}, 预测期: {pred_len}")

        # 6. 准备输入数据
        update_progress("步骤6: 准备输入数据...")
        x_df = df.loc[-lookback:, ['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.loc[-lookback:, 'timestamps'].reset_index(drop=True)

        # 生成未来日期 - 🎯 修复：只生成交易日
        last_historical_date = df['timestamps'].iloc[-1]
        future_dates = generate_trading_dates_only(last_historical_date, pred_len)

        if len(future_dates) < pred_len:
            update_progress(f"⚠️ 警告：只生成了 {len(future_dates)} 个交易日，少于请求的 {pred_len} 天")
            pred_len = len(future_dates)

        update_progress(f"输入数据形状: {x_df.shape}")
        update_progress(f"历史数据时间范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
        if future_dates:
            update_progress(f"预测时间范围: {future_dates[0]} 到 {future_dates[-1]}")

        # 7. 执行基础预测
        update_progress("步骤7: 执行基础价格预测...")
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(future_dates),
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )

        update_progress("✅ 基础预测完成")

        # 🎯 新增：对基础预测进行合理性检查
        update_progress("步骤7.2: 验证预测结果合理性...")
        historical_df_for_validation = df.loc[-lookback:].reset_index(drop=True)
        validated_pred_df = validate_prediction_results(historical_df_for_validation, pred_df)

        # 🎯 新增：对基础预测进行平滑处理
        update_progress("步骤7.5: 对预测结果进行平滑优化...")
        smoothed_pred_df = smooth_prediction_results(validated_pred_df, historical_df_for_validation)

        # 🎯 修复：应用节后调整（特别是国庆节后）
        holiday_periods = [
            {
                'start': '2025-10-01',
                'end': '2025-10-09',  # 国庆后第一个交易日（10月9日周四）
                'adjustment_days': 5,
                'effect_strength': 0.03  # 节后通常有正面效应
            }
        ]

        adjusted_pred_df = apply_post_holiday_adjustment(smoothed_pred_df, future_dates, holiday_periods)

        # 8. 使用多维度市场因素增强预测
        update_progress("步骤8: 应用多维度市场因素增强预测...")
        enhanced_pred_df, enhancement_info = enhance_prediction_with_market_factors(
            df.loc[-lookback:].reset_index(drop=True),
            adjusted_pred_df,  # 使用平滑调整后的预测结果
            stock_code,
            market_analyzer
        )

        # 将增强预测结果添加到信息中
        enhancement_info['enhanced_prediction'] = enhanced_pred_df

        # 9. 创建综合市场分析报告
        update_progress("步骤9: 创建市场分析报告...")
        market_report = create_comprehensive_market_report(enhancement_info, output_dir, stock_code)

        # 10. 生成预测图表
        update_progress("步骤10: 生成预测图表...")
        historical_df = df.loc[-lookback:].reset_index(drop=True)
        chart_path = plot_optimized_prediction_gui(
            historical_df, adjusted_pred_df, enhanced_pred_df, future_dates,
            stock_code, stock_name, output_dir, enhancement_info
        )

        # 11. 生成预测报告
        update_progress("步骤11: 生成预测报告...")
        if len(enhanced_pred_df) > 0:
            current_price = historical_df['close'].iloc[-1]
            base_predicted_price = adjusted_pred_df['close'].iloc[-1] if len(adjusted_pred_df) > 0 else current_price
            enhanced_predicted_price = enhanced_pred_df['close'].iloc[-1]

            base_change_pct = (base_predicted_price / current_price - 1) * 100
            enhanced_change_pct = (enhanced_predicted_price / current_price - 1) * 100

            # 输出预测结果
            update_result(f"\n📈 {stock_name}({stock_code}) 预测报告")
            update_result("=" * 50)
            update_result(f"当前价格: {current_price:.2f} 元")
            update_result(f"平滑预测价格: {base_predicted_price:.2f} 元 ({base_change_pct:+.2f}%)")
            update_result(f"增强预测价格: {enhanced_predicted_price:.2f} 元 ({enhanced_change_pct:+.2f}%)")
            update_result(f"市场因素调整因子: {enhancement_info['adjustment_factor']:.4f}")
            update_result(f"大盘状态: {enhancement_info['market_analysis']['market_status']}")
            update_result(f"板块共振: {enhancement_info['sector_analysis']['main_sector']['sector']}")
            update_result(f"宏观环境: 美国{enhancement_info['macro_analysis']['us_rate_cycle']['trend']}")
            update_result(f"公司评级: {enhancement_info['fundamental_analysis']['investment_rating']}")

            # 保存详细预测数据
            prediction_details = pd.DataFrame({
                '日期': future_dates,
                '平滑预测收盘价': adjusted_pred_df['close'].values if len(
                    adjusted_pred_df) > 0 else [current_price] * len(future_dates),
                '增强预测收盘价': enhanced_pred_df['close'].values,
                '预测成交量': enhanced_pred_df['volume'].values
            })

            prediction_file = os.path.join(output_dir, f'{stock_code}_comprehensive_predictions.csv')
            prediction_details.to_csv(prediction_file, index=False, encoding='utf-8-sig')
            update_progress(f"💾 详细预测数据已保存: {prediction_file}")

        update_progress(f"\n🎉 {stock_name}({stock_code}) 预测完成!")
        update_progress(f"📊 预测图表: {chart_path}")

        return True, "预测完成"

    except Exception as e:
        error_msg = f"❌ 预测过程中出现错误: {e}"
        update_result(error_msg)
        import traceback
        traceback.print_exc()
        return False, error_msg


def enhance_prediction_with_market_factors(historical_df, prediction_df, stock_code, market_analyzer):
    """
    使用市场因素增强预测结果
    """
    print("\n🎯 使用市场因素增强预测...")

    # 获取各类市场分析
    market_analysis = market_analyzer.analyze_market_trend()
    sector_analysis = market_analyzer.analyze_sector_resonance(stock_code)
    macro_analysis = market_analyzer.analyze_macro_factors()
    fundamental_analysis = market_analyzer.analyze_company_fundamentals(stock_code)

    # 计算综合调整因子
    adjustment_factor = calculate_enhanced_adjustment_factor(
        market_analysis, sector_analysis, macro_analysis, fundamental_analysis
    )

    print(f"📈 综合调整因子: {adjustment_factor:.4f}")

    # 应用调整到预测结果
    enhanced_prediction = prediction_df.copy()

    # 对价格预测进行调整
    price_columns = ['close', 'open', 'high', 'low']
    for col in price_columns:
        if col in enhanced_prediction.columns:
            enhanced_prediction[col] = enhanced_prediction[col] * adjustment_factor

    # 对成交量进行调整
    if 'volume' in enhanced_prediction.columns:
        volume_adjustment = 1 + (adjustment_factor - 1) * 0.3
        enhanced_prediction['volume'] = enhanced_prediction['volume'] * volume_adjustment

    return enhanced_prediction, {
        'market_analysis': market_analysis,
        'sector_analysis': sector_analysis,
        'macro_analysis': macro_analysis,
        'fundamental_analysis': fundamental_analysis,
        'adjustment_factor': adjustment_factor
    }


def calculate_enhanced_adjustment_factor(market_analysis, sector_analysis, macro_analysis, fundamental_analysis):
    """
    计算基于多维度市场因素的调整因子
    """
    base_factor = 1.0

    # 1. 大盘趋势影响 (权重25%)
    if market_analysis['overall_is_main_uptrend']:
        trend_strength = market_analysis['overall_trend_strength']
        base_factor *= (1 + trend_strength * 0.08)
    else:
        trend_strength = market_analysis['overall_trend_strength']
        base_factor *= (1 + (trend_strength - 0.5) * 0.04)

    # 2. 板块共振影响 (权重25%)
    resonance_score = sector_analysis['resonance_score']
    sector_count = sector_analysis['sector_count']

    if sector_analysis['is_sector_hot']:
        base_factor *= (1 + resonance_score * 0.06 + min(sector_count * 0.01, 0.03))
    else:
        base_factor *= (1 + (resonance_score - 0.5) * 0.02)

    # 3. 宏观因素影响 (权重20%)
    macro_score = macro_analysis['overall_macro_score']
    base_factor *= (1 + (macro_score - 0.5) * 0.06)

    # 4. 美国降息周期特殊影响 (权重10%)
    us_rate_trend = macro_analysis['us_rate_cycle']['trend']
    if us_rate_trend == '降息周期':
        expected_cuts = macro_analysis['us_rate_cycle']['expected_cuts_2025']
        base_factor *= (1 + expected_cuts * 0.015)

    # 5. 公司基本面影响 (权重20%)
    fundamental_score = fundamental_analysis['fundamental_score']
    base_factor *= (1 + (fundamental_score - 0.5) * 0.08)

    # 🎯 限制调整幅度在更合理范围内 (0.9 ~ 1.1)，避免过度调整
    return max(0.9, min(1.1, base_factor))


def create_comprehensive_market_report(enhancement_info, output_dir, stock_code):
    """
    创建综合市场分析报告
    """
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_code': stock_code,
        'market_analysis': enhancement_info['market_analysis'],
        'sector_analysis': enhancement_info['sector_analysis'],
        'macro_analysis': enhancement_info['macro_analysis'],
        'fundamental_analysis': enhancement_info['fundamental_analysis'],
        'adjustment_factor': enhancement_info['adjustment_factor']
    }

    # 保存报告
    report_file = os.path.join(output_dir, f'{stock_code}_comprehensive_analysis_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📋 综合分析报告已保存: {report_file}")
    return report


def plot_optimized_prediction_gui(historical_df, base_pred_df, enhanced_pred_df, future_trading_dates,
                                  stock_code, stock_name, output_dir, enhancement_info=None):
    """
    🎯 优化版：清晰显示每个交易日的预测图表
    """
    ensure_output_directory(output_dir)

    # 设置配色
    colors = {
        'historical': '#1f77b4',
        'prediction': '#ff7f0e',
        'enhanced': '#2ca02c',
        'background': '#f8f9fa',
        'grid': '#e9ecef'
    }

    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{stock_name}({stock_code}) - 优化版交易日预测图表', fontsize=16, fontweight='bold')

    # 设置背景色
    fig.patch.set_facecolor('white')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(colors['background'])

    # 🎯 优化1: 使用实际日期作为x轴，但只显示交易日
    all_dates = list(historical_df['timestamps']) + future_trading_dates

    # 1. 主价格图表
    current_price = historical_df['close'].iloc[-1]

    # 绘制历史价格
    ax1.plot(historical_df['timestamps'], historical_df['close'],
             color=colors['historical'], linewidth=2.5, label='历史价格')

    # 绘制预测价格
    if len(future_trading_dates) > 0:
        # 绘制基础预测
        ax1.plot(future_trading_dates, base_pred_df['close'],
                 color=colors['prediction'], linewidth=2, label='平滑预测', linestyle='--')

        # 绘制增强预测
        ax1.plot(future_trading_dates, enhanced_pred_df['close'],
                 color=colors['enhanced'], linewidth=2.5, label='增强预测')

        # 🎯 修复：使用更安全的关键日期标记
        mark_key_dates_safe(ax1, future_trading_dates, enhanced_pred_df)

    ax1.set_ylabel('收盘价 (元)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, color=colors['grid'], alpha=0.7)
    ax1.set_title(f'价格走势预测 - 当前价: {current_price:.2f}元', fontweight='bold', fontsize=13)

    # 🎯 优化2: 使用每周标记，避免过于密集
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))  # 每两周一个标记
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, fontsize=9)

    # 2. 成交量图表
    ax2.bar(historical_df['timestamps'], historical_df['volume'],
            alpha=0.6, color=colors['historical'], label='历史成交量')

    if len(future_trading_dates) > 0:
        ax2.bar(future_trading_dates, enhanced_pred_df['volume'],
                alpha=0.6, color=colors['enhanced'], label='预测成交量')

    ax2.set_ylabel('成交量', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, color=colors['grid'], alpha=0.7)
    ax2.set_title('成交量预测', fontweight='bold', fontsize=13)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=9)

    # 3. 价格变化率图表
    ax3.plot(historical_df['timestamps'], historical_df['close'].pct_change() * 100,
             color=colors['historical'], linewidth=1.5, label='历史涨跌幅', alpha=0.7)

    if len(future_trading_dates) > 0:
        pred_returns = enhanced_pred_df['close'].pct_change() * 100
        ax3.plot(future_trading_dates, pred_returns,
                 color=colors['enhanced'], linewidth=2, label='预测涨跌幅')

        # 添加零线参考
        ax3.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)

    ax3.set_ylabel('日涨跌幅 (%)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, color=colors['grid'], alpha=0.7)
    ax3.set_title('价格变化率分析', fontweight='bold', fontsize=13)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, fontsize=9)

    # 4. 市场因素分析
    if enhancement_info:
        factors = ['大盘趋势', '板块共振', '宏观环境', '美国降息', '基本面']
        scores = [
            enhancement_info['market_analysis']['overall_trend_strength'],
            enhancement_info['sector_analysis']['resonance_score'],
            enhancement_info['macro_analysis']['overall_macro_score'],
            0.7 if enhancement_info['macro_analysis']['us_rate_cycle']['trend'] == '降息周期' else 0.3,
            enhancement_info['fundamental_analysis']['fundamental_score']
        ]

        colors_bars = [colors['historical'], colors['prediction'], colors['enhanced'], '#f39c12', '#9b59b6']

        bars = ax4.bar(factors, scores, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('评分', fontsize=12, fontweight='bold')
        ax4.set_title('市场因素评分分析', fontweight='bold', fontsize=13)
        ax4.grid(True, alpha=0.3, axis='y')

        # 在柱状图上显示具体数值
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 添加平均线
        avg_score = np.mean(scores)
        ax4.axhline(y=avg_score, color='red', linestyle='--', alpha=0.7,
                    label=f'平均分: {avg_score:.2f}')
        ax4.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # 保存图片
    chart_filename = os.path.join(output_dir, f'{stock_code}_optimized_prediction.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📊 优化版预测图表已保存: {chart_filename}")
    return chart_filename


def mark_key_dates_safe(ax, future_dates, pred_df):
    """
    🎯 安全版：标记关键日期和价格点，避免类型错误
    """
    if len(future_dates) == 0 or len(pred_df) == 0:
        return

    try:
        # 重置索引确保使用整数索引
        pred_df_reset = pred_df.reset_index(drop=True)

        # 获取最高点和最低点的整数索引
        if hasattr(pred_df_reset['close'], 'idxmax'):
            max_idx = pred_df_reset['close'].idxmax()
            min_idx = pred_df_reset['close'].idxmin()
        else:
            # 备用方法
            max_idx = np.argmax(pred_df_reset['close'].values)
            min_idx = np.argmin(pred_df_reset['close'].values)

        # 确保索引在有效范围内
        max_idx = min(int(max_idx), len(future_dates) - 1)
        min_idx = min(int(min_idx), len(future_dates) - 1)

        # 标记最高点
        if 0 <= max_idx < len(future_dates):
            max_price = pred_df_reset['close'].iloc[max_idx]
            ax.plot(future_dates[max_idx], max_price,
                    'v', color='red', markersize=8, label=f'最高点: {max_price:.2f}')

        # 标记最低点
        if 0 <= min_idx < len(future_dates):
            min_price = pred_df_reset['close'].iloc[min_idx]
            ax.plot(future_dates[min_idx], min_price,
                    '^', color='green', markersize=8, label=f'最低点: {min_price:.2f}')

        # 标记预测结束点
        if len(future_dates) > 0:
            final_price = pred_df_reset['close'].iloc[-1]
            ax.plot(future_dates[-1], final_price,
                    's', color='blue', markersize=6, label=f'最终预测: {final_price:.2f}')

    except Exception as e:
        print(f"⚠️ 标记关键日期时出现错误: {e}")
        # 如果出错，跳过标记但不影响整体流程


# ==================== 主函数 ====================
def main():
    """主函数：启动GUI界面"""
    root = tk.Tk()
    app = StockPredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()