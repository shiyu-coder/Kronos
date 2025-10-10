#!/usr/bin/env python3
"""
基于配置文件的股票预测脚本
支持灵活的参数配置和批量处理
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import torch
import numpy as np
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# 导入自适应配置系统
try:
    from adaptive_config import AdaptiveSamplingConfig
    from enhanced_adaptive_config import EnhancedAdaptiveSamplingConfig
    ADAPTIVE_CONFIG_AVAILABLE = True
    ENHANCED_ADAPTIVE_AVAILABLE = True
except ImportError:
    try:
        from adaptive_config import AdaptiveSamplingConfig
        ADAPTIVE_CONFIG_AVAILABLE = True
        ENHANCED_ADAPTIVE_AVAILABLE = False
        print("⚠️ enhanced_adaptive_config.py 未找到，将使用基础自适应配置")
    except ImportError:
        ADAPTIVE_CONFIG_AVAILABLE = False
        ENHANCED_ADAPTIVE_AVAILABLE = False
        print("⚠️ 自适应配置模块未找到，将使用配置文件中的固定参数")

# 设置中文字体支持
import matplotlib
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 简化的字体设置方案
def setup_chinese_font():
    """设置中文字体 - 简化版本"""
    import platform
    import matplotlib.font_manager as fm
    
    try:
        # 获取系统平台
        system = platform.system()
        print(f"🖥️ 检测到系统: {system}")
        
        # 获取可用字体列表
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 根据系统选择合适的字体
        if system == "Darwin":  # macOS
            # macOS 优先字体列表
            preferred_fonts = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB']
            
            # 检查是否有 Arial Unicode MS（最可靠的中文字体）
            if 'Arial Unicode MS' in available_fonts:
                plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
                print("✅ 使用 Arial Unicode MS 字体（支持中文）")
                return True
            else:
                # 使用默认字体，不支持中文
                plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']
                print("📝 使用默认字体（不支持中文）")
                return False
                
        elif system == "Windows":
            # Windows 字体
            if 'Microsoft YaHei' in available_fonts:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                print("✅ 使用微软雅黑字体（支持中文）")
                return True
            elif 'SimHei' in available_fonts:
                plt.rcParams['font.sans-serif'] = ['SimHei']
                print("✅ 使用黑体字体（支持中文）")
                return True
            else:
                plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
                print("📝 使用默认字体（不支持中文）")
                return False
                
        else:  # Linux 和其他系统
            # Linux 字体
            linux_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC']
            for font in linux_fonts:
                if font in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font]
                    print(f"✅ 使用 {font} 字体（支持中文）")
                    return True
            
            # 使用默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            print("📝 使用默认字体（不支持中文）")
            return False
            
    except Exception as e:
        print(f"❌ 字体设置失败: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        return False

# 调用字体设置函数
USE_CHINESE_LABELS = setup_chinese_font()

if USE_CHINESE_LABELS:
    print("✅ 中文字体支持正常，将使用中文标签")
else:
    print("⚠️ 中文字体不支持，将使用英文标签")

sys.path.append("./")
from model import Kronos, KronosTokenizer, KronosPredictor
from config_manager import ConfigManager

# 标签字典 - 支持中英文切换
LABELS = {
    'chinese': {
        'price_prediction_results': '价格预测结果',
        'future_price_prediction': '未来价格预测',
        'actual_price': '实际价格',
        'predicted_price': '预测价格',
        'historical_price': '历史价格',
        'historical_data': '历史数据',
        'prediction_start': '预测起点',
        'close_price_comparison': '收盘价对比',
        'price': '价格',
        'actual_volume': '实际成交量',
        'predicted_volume': '预测成交量',
        'historical_volume': '历史成交量',
        'volume_comparison': '成交量对比',
        'volume': '成交量',
        'price_prediction_error': '价格预测误差',
        'absolute_error': '绝对误差',
        'price_forecast': '价格预测',
        'volume_forecast': '成交量预测',
        'price_change_rate': '价格变化率 (%)',
        'return_rate': '收益率 (%)',
        'historical_returns': '历史收益率',
        'predicted_returns': '预测收益率',
        'performance_analysis': '性能分析',
        'prediction_analysis': '预测分析',
        'prediction_performance_metrics': '预测性能指标',
        'accuracy_measures': '准确性指标:',
        'mae': '平均绝对误差 (MAE)',
        'rmse': '均方根误差 (RMSE)',
        'mape': '平均绝对百分比误差',
        'direction_accuracy': '方向准确率',
        'price_correlation': '价格相关性',
        'price_range_analysis': '价格区间分析:',
        'actual_range': '实际价格区间',
        'predicted_range': '预测价格区间',
        'volatility_analysis': '波动性分析:',
        'actual_volatility': '实际波动性',
        'predicted_volatility': '预测波动性',
        'volatility_capture': '波动性捕获',
        'model_assessment': '模型评估:',
        'overall_grade': '综合评级',
        'trend_capture': '趋势捕获',
        'good': '良好',
        'fair': '一般',
        'poor': '较差',
        'future_prediction_analysis': '未来预测分析',
        'current_market_status': '当前市场状态:',
        'current_price': '当前价格',
        'data_period': '数据周期',
        'historical_volatility': '历史波动性',
        'prediction_results': '预测结果:',
        'predicted_final_price': '预测最终价格',
        'price_change': '价格变化',
        'trend_direction': '趋势方向',
        'predicted_range_section': '预测区间:',
        'maximum': '最高价',
        'minimum': '最低价',
        'price_range': '价格区间',
        'risk_assessment': '风险评估:',
        'volatility_ratio': '波动性比率',
        'risk_level': '风险等级',
        'confidence': '置信度',
        'high': '高',
        'medium': '中',
        'low': '低'
    },
    'english': {
        'price_prediction_results': 'Price Prediction Results',
        'future_price_prediction': 'Future Price Prediction',
        'actual_price': 'Actual Price',
        'predicted_price': 'Predicted Price',
        'historical_price': 'Historical Price',
        'historical_data': 'Historical Data',
        'prediction_start': 'Prediction Start',
        'close_price_comparison': 'Close Price Comparison',
        'price': 'Price',
        'actual_volume': 'Actual Volume',
        'predicted_volume': 'Predicted Volume',
        'historical_volume': 'Historical Volume',
        'volume_comparison': 'Volume Comparison',
        'volume': 'Volume',
        'price_prediction_error': 'Price Prediction Error',
        'absolute_error': 'Absolute Error',
        'price_forecast': 'Price Forecast',
        'volume_forecast': 'Volume Forecast',
        'price_change_rate': 'Price Change Rate (%)',
        'return_rate': 'Return (%)',
        'historical_returns': 'Historical Returns',
        'predicted_returns': 'Predicted Returns',
        'performance_analysis': 'Performance Analysis',
        'prediction_analysis': 'Prediction Analysis',
        'prediction_performance_metrics': 'PREDICTION PERFORMANCE METRICS',
        'accuracy_measures': 'Accuracy Measures:',
        'mae': 'Mean Absolute Error (MAE)',
        'rmse': 'Root Mean Square Error (RMSE)',
        'mape': 'Mean Absolute Percentage Error',
        'direction_accuracy': 'Direction Accuracy',
        'price_correlation': 'Price Correlation',
        'price_range_analysis': 'Price Range Analysis:',
        'actual_range': 'Actual Range',
        'predicted_range': 'Predicted Range',
        'volatility_analysis': 'Volatility Analysis:',
        'actual_volatility': 'Actual Volatility',
        'predicted_volatility': 'Predicted Volatility',
        'volatility_capture': 'Volatility Capture',
        'model_assessment': 'Model Assessment:',
        'overall_grade': 'Overall Grade',
        'trend_capture': 'Trend Capture',
        'good': 'Good',
        'fair': 'Fair',
        'poor': 'Poor',
        'future_prediction_analysis': 'FUTURE PREDICTION ANALYSIS',
        'current_market_status': 'Current Market Status:',
        'current_price': 'Current Price',
        'data_period': 'Data Period',
        'historical_volatility': 'Historical Volatility',
        'prediction_results': 'Prediction Results:',
        'predicted_final_price': 'Predicted Final Price',
        'price_change': 'Price Change',
        'trend_direction': 'Trend Direction',
        'predicted_range_section': 'Predicted Range:',
        'maximum': 'Maximum',
        'minimum': 'Minimum',
        'price_range': 'Range',
        'risk_assessment': 'Risk Assessment:',
        'volatility_ratio': 'Volatility Ratio',
        'risk_level': 'Risk Level',
        'confidence': 'Confidence',
        'high': 'High',
        'medium': 'Medium',
        'low': 'Low'
    }
}

# 选择标签语言
CURRENT_LABELS = LABELS['chinese'] if USE_CHINESE_LABELS else LABELS['english']

def get_label(key: str) -> str:
    """获取标签文本"""
    return CURRENT_LABELS.get(key, key)

def get_text_font():
    """获取当前设置的字体，用于文本显示"""
    current_font = plt.rcParams['font.sans-serif'][0]
    # 如果支持中文，使用当前字体；否则使用默认字体避免警告
    if USE_CHINESE_LABELS:
        return current_font
    else:
        # 对于不支持中文的情况，使用默认字体并避免 monospace
        return 'DejaVu Sans'

def setup_matplotlib_backend():
    """设置 matplotlib 后端以减少警告"""
    try:
        import matplotlib
        # 如果不支持中文，设置一些参数来减少警告
        if not USE_CHINESE_LABELS:
            # 禁用一些可能产生警告的功能
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")
    except Exception:
        pass

# 调用后端设置
setup_matplotlib_backend()

class StockPredictor:
    """股票预测器"""
    
    def __init__(self, config_manager: ConfigManager, use_adaptive: bool = False):
        """
        初始化预测器
        
        Args:
            config_manager (ConfigManager): 配置管理器
            use_adaptive (bool): 是否使用自适应采样参数
        """
        self.config = config_manager
        self.use_adaptive = use_adaptive
        self.adaptive_config = None
        
        # 初始化自适应配置系统
        if self.use_adaptive and ENHANCED_ADAPTIVE_AVAILABLE:
            self.adaptive_config = EnhancedAdaptiveSamplingConfig()
            self.enhanced_adaptive = True
            print("✅ 增强版自适应采样参数系统已启用 (考虑时间间隔)")
        elif self.use_adaptive and ADAPTIVE_CONFIG_AVAILABLE:
            self.adaptive_config = AdaptiveSamplingConfig()
            self.enhanced_adaptive = False
            print("✅ 基础自适应采样参数系统已启用")
        elif self.use_adaptive:
            print("⚠️ 自适应配置不可用，将使用配置文件中的固定参数")
            self.use_adaptive = False
            self.enhanced_adaptive = False
        else:
            self.enhanced_adaptive = False
        
        self.setup_matplotlib()
        self.setup_directories()
        
        # 加载模型
        self.tokenizer = None
        self.model = None
        self.predictor = None
        self.device = None
        
    def setup_matplotlib(self):
        """设置 matplotlib"""
        chart_config = self.config.get_chart_config()
        fonts = chart_config.get('fonts', ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'])
        plt.rcParams['font.sans-serif'] = fonts
        plt.rcParams['axes.unicode_minus'] = False
        
    def setup_directories(self):
        """创建必要的目录"""
        data_config = self.config.get_data_config()
        output_config = self.config.get_output_config()
        
        os.makedirs(data_config.get('output_dir', 'data'), exist_ok=True)
        os.makedirs(output_config.get('results_dir', 'results'), exist_ok=True)
    
    def load_model(self):
        """加载 Kronos 模型"""
        if self.predictor is not None:
            return  # 已经加载
            
        model_config = self.config.get_model_config()
        
        logging.info("📦 加载 Kronos 模型...")
        self.tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_name'])
        self.model = Kronos.from_pretrained(model_config['model_name'])
        
        # 选择设备
        device_config = model_config.get('device', 'auto')
        if device_config == 'auto':
            if torch.backends.mps.is_available():
                self.device = "mps"
                logging.info("✅ 使用 Apple GPU (MPS) 加速")
            elif torch.cuda.is_available():
                self.device = "cuda:0"
                logging.info("✅ 使用 NVIDIA GPU (CUDA) 加速")
            else:
                self.device = "cpu"
                logging.info("✅ 使用 CPU 计算")
        else:
            self.device = device_config
            logging.info(f"✅ 使用指定设备: {self.device}")
        
        self.predictor = KronosPredictor(
            self.model, 
            self.tokenizer, 
            device=self.device, 
            max_context=model_config.get('max_context', 512)
        )
    
    def download_data(self, symbol: str) -> Optional[str]:
        """
        下载股票数据
        
        Args:
            symbol (str): 股票代码
            
        Returns:
            Optional[str]: 数据文件路径，失败时返回 None
        """
        data_config = self.config.get_data_config()
        
        if USE_CHINESE_LABELS:
            logging.info(f"开始下载 {symbol} 数据...")
        else:
            logging.info(f"Downloading {symbol} data...")
        logging.info(f"   时间周期: {data_config['period']}")
        logging.info(f"   时间间隔: {data_config['interval']}")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=data_config['period'], interval=data_config['interval'])
            
            if data.empty:
                logging.error(f"❌ 无法获取 {symbol} 数据")
                return None
            
            # 数据处理
            data = data.reset_index()
            column_mapping = {
                'Datetime': 'timestamps',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            data = data.rename(columns=column_mapping)
            data['amount'] = data['close'] * data['volume']
            
            # 保存数据
            filename = f"{symbol}_{data_config['interval']}_{data_config['period']}.csv"
            filepath = os.path.join(data_config['output_dir'], filename)
            data.to_csv(filepath, index=False)
            
            logging.info(f"✅ {symbol} 数据已保存: {filepath}")
            logging.info(f"   数据条数: {len(data)}")
            logging.info(f"   价格范围: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
            
            return filepath
            
        except Exception as e:
            logging.error(f"❌ 下载 {symbol} 数据失败: {e}")
            return None
    
    def predict_stock(self, symbol: str, data_file: str) -> Optional[pd.DataFrame]:
        """
        预测股票价格
        
        Args:
            symbol (str): 股票代码
            data_file (str): 数据文件路径
            
        Returns:
            Optional[pd.DataFrame]: 预测结果，失败时返回 None
        """
        try:
            # 确保模型已加载
            self.load_model()
            
            pred_config = self.config.get_prediction_config()
            
            logging.info(f"🚀 开始预测 {symbol}...")
            
            # 加载数据
            df = pd.read_csv(data_file)
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            logging.info(f"   数据总量: {len(df)} 条")
            logging.info(f"   时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
            
            # 数据划分 - 根据配置选择历史数据窗口
            lookback = pred_config['lookback']
            pred_len = pred_config['pred_len']
            start_index = pred_config.get('start_index', -1)  # 默认从最新数据开始
            
            if len(df) < lookback:
                logging.error(f"❌ {symbol} 数据不足，需要至少 {lookback} 条历史数据")
                return None
            
            # 计算起始索引
            logging.info(f"   配置的起始索引: {start_index}")
            if start_index == -1:
                # 从最新数据开始 (默认行为)
                start_idx = len(df) - lookback
                logging.info(f"   使用最新数据模式: {len(df)} - {lookback} = {start_idx}")
            elif start_index < -1:
                # 从倒数第N个位置开始，取lookback个数据
                # 例如: start_index=-400, 表示从倒数第400个位置开始取400个数据
                start_idx = len(df) + start_index
                logging.info(f"   从倒数第{abs(start_index)}个位置开始: {len(df)} + ({start_index}) = {start_idx}")
                if start_idx < 0:
                    start_idx = 0
                    logging.warning(f"⚠️ 起始索引 {start_index} 超出数据范围，调整为 0")
            else:
                # 从指定索引开始 (正数或0)
                start_idx = start_index
                logging.info(f"   使用指定索引: {start_idx}")
            
            # 确保索引范围有效
            if start_idx < 0:
                start_idx = 0
            elif start_idx + lookback > len(df):
                start_idx = len(df) - lookback
                logging.warning(f"⚠️ 调整起始索引到 {start_idx}，确保有足够的历史数据")
            
            # 获取指定长度的历史数据
            end_idx = start_idx + lookback
            x_df = df.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = df.iloc[start_idx:end_idx]['timestamps']
            
            logging.info(f"   历史数据窗口: 索引 {start_idx} 到 {end_idx-1} (共 {len(x_df)} 条)")
            logging.info(f"   历史数据时间: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
            
            # 生成未来时间戳（基于历史数据窗口的最后一个时间点）
            last_timestamp = x_timestamp.iloc[-1]
            data_config = self.config.get_data_config()
            interval = data_config['interval']
            
            # 计算时间间隔（分钟）
            if 'm' in interval:
                interval_minutes = int(interval.replace('m', ''))
            elif 'h' in interval:
                interval_minutes = int(interval.replace('h', '')) * 60
            elif 'd' in interval:
                interval_minutes = int(interval.replace('d', '')) * 24 * 60
            else:
                interval_minutes = 30  # 默认30分钟
            
            # 生成未来时间戳
            future_timestamps = []
            for i in range(1, pred_len + 1):
                future_time = last_timestamp + pd.Timedelta(minutes=interval_minutes * i)
                future_timestamps.append(future_time)
            
            y_timestamp = pd.Series(future_timestamps)
            
            # 计算预测时长
            data_config = self.config.get_data_config()
            interval = data_config['interval']
            if 'm' in interval:
                minutes = int(interval.replace('m', ''))
                hours = pred_len * minutes / 60
            elif 'h' in interval:
                hours = pred_len * int(interval.replace('h', ''))
            else:
                hours = pred_len  # 默认按小时计算
            
            logging.info(f"   历史数据: {len(x_df)} 条")
            logging.info(f"   预测长度: {pred_len} 条 (约 {hours:.1f} 小时)")
            
            # 获取采样参数（自适应或配置文件）
            if self.use_adaptive and self.adaptive_config:
                # 获取数据配置中的时间间隔
                data_config = self.config.get_data_config()
                interval = data_config.get('interval', '1h')
                
                if self.enhanced_adaptive:
                    # 使用增强版自适应参数（考虑时间间隔和数据特征）
                    adaptive_params = self.adaptive_config.get_enhanced_sampling_config(
                        symbol, interval, x_df
                    )
                    temperature = adaptive_params['temperature']
                    top_p = adaptive_params['top_p']
                    sample_count = adaptive_params['sample_count']
                    
                    logging.info(f"🎯 使用增强版自适应采样参数:")
                    logging.info(f"   标的类型: {adaptive_params['description']}")
                    logging.info(f"   时间间隔: {interval} ({adaptive_params['interval_category']})")
                    logging.info(f"   波动性因子: {adaptive_params['analysis']['volatility_factor']:.2f}")
                    logging.info(f"   实际波动性: {adaptive_params['analysis']['actual_volatility']:.2f}%" if isinstance(adaptive_params['analysis']['actual_volatility'], (int, float)) else f"   实际波动性: {adaptive_params['analysis']['actual_volatility']}")
                    logging.info(f"   Temperature: {temperature}")
                    logging.info(f"   Top_p: {top_p}")
                    logging.info(f"   Sample_count: {sample_count}")
                else:
                    # 使用基础自适应参数
                    adaptive_params = self.adaptive_config.get_sampling_config(symbol)
                    temperature = adaptive_params['temperature']
                    top_p = adaptive_params['top_p']
                    sample_count = adaptive_params['sample_count']
                    
                    logging.info(f"🎯 使用基础自适应采样参数:")
                    logging.info(f"   标的类型: {adaptive_params['description']}")
                    logging.info(f"   Temperature: {temperature}")
                    logging.info(f"   Top_p: {top_p}")
                    logging.info(f"   Sample_count: {sample_count}")
            else:
                # 使用配置文件中的固定参数
                temperature = pred_config.get('temperature', 1.0)
                top_p = pred_config.get('top_p', 0.9)
                sample_count = pred_config.get('sample_count', 1)
                
                logging.info(f"📋 使用配置文件参数:")
                logging.info(f"   Temperature: {temperature}")
                logging.info(f"   Top_p: {top_p}")
                logging.info(f"   Sample_count: {sample_count}")
            
            # 进行预测
            logging.info(f"🔮 开始预测 {symbol} 价格...")
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
                verbose=pred_config.get('verbose', True)
            )
            
            # 保存预测结果
            output_config = self.config.get_output_config()
            if output_config.get('save_predictions', True):
                results_dir = output_config.get('results_dir', 'results')
                pred_file = os.path.join(results_dir, f"{symbol}_predictions.csv")
                pred_df.to_csv(pred_file)
                logging.info(f"💾 预测结果已保存: {pred_file}")
            
            # 生成图表（两种模式）
            if start_index == -1:
                self.create_future_prediction_chart(symbol, df, pred_df, start_idx)
            else:
                self.create_prediction_chart(symbol, df, pred_df, start_idx, lookback, pred_len)
            # self.create_future_prediction_chart(symbol, df, pred_df, start_idx)
            # self.create_future_prediction_chart(symbol, df, pred_df, start_idx)
            
            return pred_df
            
        except Exception as e:
            logging.error(f"❌ 预测 {symbol} 失败: {e}")
            return None
    
    def create_prediction_chart(self, symbol: str, df: pd.DataFrame, pred_df: pd.DataFrame, 
                              start_idx: int, lookback: int, pred_len: int):
        """
        创建预测图表（支持统一时间轴缩放）
        
        Args:
            symbol (str): 股票代码
            df (pd.DataFrame): 原始数据
            pred_df (pd.DataFrame): 预测数据
            start_idx (int): 历史数据起始索引
            lookback (int): 历史数据长度
            pred_len (int): 预测长度
        """
        try:
            chart_config = self.config.get_chart_config()
            colors = chart_config.get('colors', {})
            
            # 获取真实数据用于对比（从历史数据结束位置开始）
            end_idx = start_idx + lookback
            available_true_data = df.iloc[end_idx:]
            
            if len(available_true_data) < pred_len:
                logging.warning(f"⚠️ 真实数据不足：需要 {pred_len} 个数据点，但只有 {len(available_true_data)} 个")
                # 调整预测数据长度以匹配可用的真实数据
                actual_pred_len = len(available_true_data)
                true_df = available_true_data
                
                # 截断预测数据并重新生成对应的时间戳
                pred_values = pred_df.iloc[:actual_pred_len]
                
                # 生成对应的时间戳：从历史数据结束点开始
                last_timestamp = df.iloc[end_idx-1]['timestamps']
                
                # 获取时间间隔
                data_config = self.config.get_data_config()
                interval = data_config['interval']
                if 'h' in interval:
                    interval_minutes = int(interval.replace('h', '')) * 60
                elif 'm' in interval:
                    interval_minutes = int(interval.replace('m', ''))
                else:
                    interval_minutes = 60  # 默认1小时
                
                # 生成新的时间戳
                new_timestamps = []
                for i in range(1, actual_pred_len + 1):
                    future_time = last_timestamp + pd.Timedelta(minutes=interval_minutes * i)
                    new_timestamps.append(future_time)
                
                # 重新创建预测DataFrame
                pred_df = pd.DataFrame({
                    'open': pred_values['open'].values,
                    'high': pred_values['high'].values,
                    'low': pred_values['low'].values,
                    'close': pred_values['close'].values,
                    'volume': pred_values['volume'].values,
                    'amount': pred_values['amount'].values
                }, index=pd.Series(new_timestamps))
                
                logging.info(f"   调整对比长度为: {actual_pred_len} 个数据点")
                logging.info(f"   预测时间范围: {new_timestamps[0]} 到 {new_timestamps[-1]}")
            else:
                true_df = available_true_data.iloc[:pred_len]
            
            # 获取历史数据用于显示
            hist_df = df.iloc[start_idx:end_idx].copy()
            
            # 检查是否启用统一时间轴
            use_unified_axis = chart_config.get('use_unified_time_axis', True)
            
            if use_unified_axis:
                # 创建统一时间轴数据
                hist_x, hist_y, true_x, true_y, pred_x, pred_y, time_labels, time_positions = \
                    self.create_unified_time_axis(hist_df, true_df, pred_df)
                
                hist_vol_x, hist_vol, true_vol_x, true_vol, pred_vol_x, pred_vol = \
                    self.create_unified_volume_axis(hist_df, true_df, pred_df)
            
            # 创建图表
            fig_size = chart_config.get('figure_size', [16, 12])
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=fig_size)
            
            if use_unified_axis:
                fig.suptitle(f'{symbol} {get_label("price_prediction_results")} (统一时间轴)', fontsize=18, fontweight='bold')
            else:
                fig.suptitle(f'{symbol} {get_label("price_prediction_results")}', fontsize=18, fontweight='bold')
            
            if use_unified_axis:
                # 价格对比 - 使用统一时间轴
                # 1. 绘制历史数据（用于训练的数据）
                ax1.plot(hist_x, hist_y, 
                        label=get_label('historical_data'), 
                        color='gray', 
                        linewidth=2, alpha=0.6)
                
                # 2. 绘制真实数据（用于对比的数据）
                ax1.plot(true_x, true_y, 
                        label=get_label('actual_price'), 
                        color=colors.get('actual_price', '#f7931a'), 
                        linewidth=3, alpha=0.8)
                
                # 3. 绘制预测数据
                ax1.plot(pred_x, pred_y, 
                        label=get_label('predicted_price'), 
                        color=colors.get('predicted_price', 'red'), 
                        linewidth=2, linestyle='--', alpha=0.9)
                
                # 添加分界线
                if len(hist_df) > 0:
                    ax1.axvline(x=len(hist_df)-0.5, color='black', linestyle=':', alpha=0.5, 
                               label=get_label('prediction_start'))
                
                # 设置X轴标签
                ax1.set_xticks(time_positions)
                ax1.set_xticklabels(time_labels, rotation=45)
            else:
                # 价格对比 - 使用传统时间轴
                # 1. 绘制历史数据（用于训练的数据）
                ax1.plot(hist_df['timestamps'], hist_df['close'], 
                        label=get_label('historical_data'), 
                        color='gray', 
                        linewidth=2, alpha=0.6)
                
                # 2. 绘制真实数据（用于对比的数据）
                ax1.plot(true_df['timestamps'], true_df['close'], 
                        label=get_label('actual_price'), 
                        color=colors.get('actual_price', '#f7931a'), 
                        linewidth=3, alpha=0.8)
                
                # 3. 绘制预测数据
                ax1.plot(pred_df.index, pred_df['close'], 
                        label=get_label('predicted_price'), 
                        color=colors.get('predicted_price', 'red'), 
                        linewidth=2, linestyle='--', alpha=0.9)
                
                ax1.tick_params(axis='x', rotation=45)
            
            ax1.set_title(get_label('close_price_comparison'), fontsize=14, fontweight='bold')
            ax1.set_ylabel(get_label('price'), fontsize=12)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            if use_unified_axis:
                # 成交量对比 - 使用统一时间轴
                # 1. 绘制历史成交量
                ax2.plot(hist_vol_x, hist_vol, 
                        label=get_label('historical_volume'), 
                        color='gray', 
                        linewidth=2, alpha=0.6)
                
                # 2. 绘制真实成交量
                ax2.plot(true_vol_x, true_vol, 
                        label=get_label('actual_volume'), 
                        color=colors.get('actual_volume', 'blue'), 
                        linewidth=2, alpha=0.7)
                
                # 3. 绘制预测成交量
                ax2.plot(pred_vol_x, pred_vol, 
                        label=get_label('predicted_volume'), 
                        color=colors.get('predicted_volume', 'red'), 
                        linewidth=2, linestyle='--', alpha=0.9)
                
                # 添加分界线
                if len(hist_df) > 0:
                    ax2.axvline(x=len(hist_df)-0.5, color='black', linestyle=':', alpha=0.5)
                
                # 设置X轴标签
                ax2.set_xticks(time_positions)
                ax2.set_xticklabels(time_labels, rotation=45)
            else:
                # 成交量对比 - 使用传统时间轴
                # 1. 绘制历史成交量
                ax2.plot(hist_df['timestamps'], hist_df['volume'], 
                        label=get_label('historical_volume'), 
                        color='gray', 
                        linewidth=2, alpha=0.6)
                
                # 2. 绘制真实成交量
                ax2.plot(true_df['timestamps'], true_df['volume'], 
                        label=get_label('actual_volume'), 
                        color=colors.get('actual_volume', 'blue'), 
                        linewidth=2, alpha=0.7)
                
                # 3. 绘制预测成交量
                ax2.plot(pred_df.index, pred_df['volume'], 
                        label=get_label('predicted_volume'), 
                        color=colors.get('predicted_volume', 'red'), 
                        linewidth=2, linestyle='--', alpha=0.9)
                
                ax2.tick_params(axis='x', rotation=45)
            
            ax2.set_title(get_label('volume_comparison'), fontsize=14, fontweight='bold')
            ax2.set_ylabel(get_label('volume'), fontsize=12)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            if use_unified_axis:
                # 价格误差 - 使用统一时间轴
                min_len = min(len(true_y), len(pred_y))
                if min_len > 0:
                    price_error = abs(true_y[:min_len] - pred_y[:min_len])
                    error_x = true_x[:min_len]
                    
                    ax3.plot(error_x, price_error, 
                            color=colors.get('error_color', 'orange'), 
                            linewidth=2, alpha=0.8)
                    ax3.fill_between(error_x, price_error, alpha=0.3, 
                                   color=colors.get('error_color', 'orange'))
                    
                    # 设置X轴标签
                    ax3.set_xticks(time_positions)
                    ax3.set_xticklabels(time_labels, rotation=45)
            else:
                # 价格误差 - 使用传统时间轴
                price_error = abs(true_df['close'].values - pred_df['close'].values)
                ax3.plot(true_df['timestamps'], price_error, 
                        color=colors.get('error_color', 'orange'), 
                        linewidth=2, alpha=0.8)
                ax3.fill_between(true_df['timestamps'], price_error, alpha=0.3, 
                               color=colors.get('error_color', 'orange'))
                ax3.tick_params(axis='x', rotation=45)
            
            ax3.set_title(get_label('price_prediction_error'), fontsize=14, fontweight='bold')
            ax3.set_ylabel(get_label('absolute_error'), fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # 性能指标
            self.add_performance_metrics(ax4, true_df, pred_df)
            
            plt.tight_layout()
            
            # 保存图表
            output_config = self.config.get_output_config()
            results_dir = output_config.get('results_dir', 'results')
            save_format = chart_config.get('save_format', 'png')
            dpi = chart_config.get('dpi', 300)
            
            chart_file = os.path.join(results_dir, f"{symbol}_prediction_chart.{save_format}")
            plt.savefig(chart_file, dpi=dpi, bbox_inches='tight', facecolor='white')
            logging.info(f"📊 图表已保存: {chart_file}")
            
            # 显示图表
            if output_config.get('show_plots', True):
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logging.error(f"❌ 创建 {symbol} 图表失败: {e}")
    
    def create_future_prediction_chart(self, symbol: str, df: pd.DataFrame, pred_df: pd.DataFrame, start_idx: int):
        """
        创建未来预测图表（没有真实数据对比，支持统一时间轴）
        
        Args:
            symbol (str): 股票代码
            df (pd.DataFrame): 历史数据
            pred_df (pd.DataFrame): 预测数据
            start_idx (int): 历史数据开始索引
        """
        try:
            chart_config = self.config.get_chart_config()
            colors = chart_config.get('colors', {})
            
            # 获取用于预测的历史数据
            historical_df = df.iloc[start_idx:]
            
            # 检查是否启用统一时间轴
            use_unified_axis = chart_config.get('use_unified_time_axis', True)
            
            if use_unified_axis:
                # 创建统一时间轴数据（未来预测模式）
                hist_len = len(historical_df)
                pred_len = len(pred_df)
            
                # 历史数据的X轴位置
                hist_x = np.arange(hist_len)
                hist_y = historical_df['close'].values
                hist_vol = historical_df['volume'].values
                
                # 预测数据的X轴位置（从历史数据结束点开始）
                pred_x = np.arange(hist_len, hist_len + pred_len)
                pred_y = pred_df['close'].values
                pred_vol = pred_df['volume'].values
                
                # 创建时间标签
                time_positions = []
                time_labels = []
                
                # 历史数据的起始和结束时间
                if hist_len > 0:
                    time_positions.extend([0, hist_len - 1])
                    time_labels.extend([
                        historical_df['timestamps'].iloc[0].strftime('%m-%d %H:%M'),
                        historical_df['timestamps'].iloc[-1].strftime('%m-%d %H:%M')
                    ])
                
                # 预测数据的结束时间
                if pred_len > 0:
                    time_positions.append(hist_len + pred_len - 1)
                    if hasattr(pred_df.index, 'strftime'):
                        time_labels.append(pred_df.index[-1].strftime('%m-%d %H:%M'))
                    else:
                        time_labels.append('预测终点')
            
            # 创建图表
            fig_size = chart_config.get('figure_size', [16, 12])
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=fig_size)
            
            if use_unified_axis:
                fig.suptitle(f'{symbol} {get_label("future_price_prediction")} (统一时间轴)', fontsize=18, fontweight='bold')
            else:
                fig.suptitle(f'{symbol} {get_label("future_price_prediction")}', fontsize=18, fontweight='bold')
            
            if use_unified_axis:
                # 历史价格 + 预测价格 - 使用统一时间轴
                ax1.plot(hist_x, hist_y, 
                        label=get_label('historical_price'), 
                        color=colors.get('actual_price', '#f7931a'), 
                        linewidth=2, alpha=0.8)
                ax1.plot(pred_x, pred_y, 
                        label=get_label('predicted_price'), 
                        color=colors.get('predicted_price', 'red'), 
                        linewidth=2, linestyle='--', alpha=0.9)
                
                # 添加分界线
                if hist_len > 0:
                    ax1.axvline(x=hist_len-0.5, color='gray', linestyle=':', alpha=0.7, 
                               label=get_label('prediction_start'))
                
                ax1.set_xticks(time_positions)
                ax1.set_xticklabels(time_labels, rotation=45)
            else:
                # 历史价格 + 预测价格 - 使用传统时间轴
                ax1.plot(historical_df['timestamps'], historical_df['close'], 
                        label=get_label('historical_price'), 
                        color=colors.get('actual_price', '#f7931a'), 
                        linewidth=2, alpha=0.8)
                ax1.plot(pred_df.index, pred_df['close'], 
                        label=get_label('predicted_price'), 
                        color=colors.get('predicted_price', 'red'), 
                        linewidth=2, linestyle='--', alpha=0.9)
                ax1.axvline(x=historical_df['timestamps'].iloc[-1], color='gray', linestyle=':', alpha=0.7, label=get_label('prediction_start'))
                ax1.tick_params(axis='x', rotation=45)
            
            ax1.set_title(get_label('price_forecast'), fontsize=14, fontweight='bold')
            ax1.set_ylabel(get_label('price'), fontsize=12)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            if use_unified_axis:
                # 历史成交量 + 预测成交量 - 使用统一时间轴
                ax2.plot(hist_x, hist_vol, 
                        label=get_label('historical_volume'), 
                        color=colors.get('actual_volume', 'blue'), 
                        linewidth=2, alpha=0.7)
                ax2.plot(pred_x, pred_vol, 
                        label=get_label('predicted_volume'), 
                        color=colors.get('predicted_volume', 'red'), 
                        linewidth=2, linestyle='--', alpha=0.9)
                
                # 添加分界线
                if hist_len > 0:
                    ax2.axvline(x=hist_len-0.5, color='gray', linestyle=':', alpha=0.7)
                
                ax2.set_xticks(time_positions)
                ax2.set_xticklabels(time_labels, rotation=45)
            else:
                # 历史成交量 + 预测成交量 - 使用传统时间轴
                ax2.plot(historical_df['timestamps'], historical_df['volume'], 
                        label=get_label('historical_volume'), 
                        color=colors.get('actual_volume', 'blue'), 
                        linewidth=2, alpha=0.7)
                ax2.plot(pred_df.index, pred_df['volume'], 
                        label=get_label('predicted_volume'), 
                        color=colors.get('predicted_volume', 'red'), 
                        linewidth=2, linestyle='--', alpha=0.9)
                ax2.axvline(x=historical_df['timestamps'].iloc[-1], color='gray', linestyle=':', alpha=0.7)
                ax2.tick_params(axis='x', rotation=45)
            
            ax2.set_title(get_label('volume_forecast'), fontsize=14, fontweight='bold')
            ax2.set_ylabel(get_label('volume'), fontsize=12)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            if use_unified_axis:
                # 价格变化趋势 - 使用统一时间轴
                historical_returns = np.diff(hist_y) / hist_y[:-1] * 100 if len(hist_y) > 1 else np.array([])
                predicted_returns = np.diff(pred_y) / pred_y[:-1] * 100 if len(pred_y) > 1 else np.array([])
                
                if len(historical_returns) > 0:
                    ax3.plot(hist_x[1:], historical_returns, 
                            color='blue', linewidth=1, alpha=0.7, label=get_label('historical_returns'))
                
                if len(predicted_returns) > 0:
                    ax3.plot(pred_x[1:], predicted_returns, 
                            color='red', linewidth=2, alpha=0.9, label=get_label('predicted_returns'))
                
                # 添加分界线和零线
                if hist_len > 0:
                    ax3.axvline(x=hist_len-0.5, color='gray', linestyle=':', alpha=0.7)
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                ax3.set_xticks(time_positions)
                ax3.set_xticklabels(time_labels, rotation=45)
            else:
                # 价格变化趋势 - 使用传统时间轴
                historical_returns = historical_df['close'].pct_change().fillna(0)
                predicted_returns = pred_df['close'].pct_change().fillna(0)
                
                ax3.plot(historical_df['timestamps'], historical_returns * 100, 
                        color='blue', linewidth=1, alpha=0.7, label=get_label('historical_returns'))
                ax3.plot(pred_df.index, predicted_returns * 100, 
                        color='red', linewidth=2, alpha=0.9, label=get_label('predicted_returns'))
                ax3.axvline(x=historical_df['timestamps'].iloc[-1], color='gray', linestyle=':', alpha=0.7)
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.tick_params(axis='x', rotation=45)
            
            ax3.set_title(get_label('price_change_rate'), fontsize=14, fontweight='bold')
            ax3.set_ylabel(get_label('return_rate'), fontsize=12)
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
            
            # 预测统计信息
            self.add_future_prediction_stats(ax4, historical_df, pred_df)
            
            plt.tight_layout()
            
            # 保存图表
            output_config = self.config.get_output_config()
            results_dir = output_config.get('results_dir', 'results')
            save_format = chart_config.get('save_format', 'png')
            dpi = chart_config.get('dpi', 300)
            
            chart_file = os.path.join(results_dir, f"{symbol}_future_prediction.{save_format}")
            plt.savefig(chart_file, dpi=dpi, bbox_inches='tight', facecolor='white')
            logging.info(f"📊 未来预测图表已保存: {chart_file}")
            
            # 显示图表
            if output_config.get('show_plots', True):
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logging.error(f"❌ 创建 {symbol} 未来预测图表失败: {e}")
    
    def add_future_prediction_stats(self, ax, historical_df: pd.DataFrame, pred_df: pd.DataFrame):
        """添加未来预测统计信息"""
        # 计算统计指标
        current_price = historical_df['close'].iloc[-1]
        predicted_final_price = pred_df['close'].iloc[-1]
        price_change = predicted_final_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        predicted_max = pred_df['close'].max()
        predicted_min = pred_df['close'].min()
        predicted_volatility = pred_df['close'].std()
        
        historical_volatility = historical_df['close'].std()
        
        # 预测趋势
        if price_change > 0:
            if USE_CHINESE_LABELS:
                trend = "↗ 上涨"  # 使用更兼容的箭头符号
            else:
                trend = "↗ Upward"
            trend_color = "green"
        else:
            if USE_CHINESE_LABELS:
                trend = "↘ 下跌"  # 使用更兼容的箭头符号
            else:
                trend = "↘ Downward"
            trend_color = "red"
        
        # 统计信息文本
        stats_text = f"""{get_label('future_prediction_analysis')}

{get_label('current_market_status')}
• {get_label('current_price')}: ${current_price:,.2f}
• {get_label('data_period')}: {historical_df['timestamps'].iloc[0].strftime('%Y-%m-%d')} 至 {historical_df['timestamps'].iloc[-1].strftime('%Y-%m-%d')}
• {get_label('historical_volatility')}: ${historical_volatility:,.2f}

{get_label('prediction_results')}
• {get_label('predicted_final_price')}: ${predicted_final_price:,.2f}
• {get_label('price_change')}: ${price_change:+,.2f} ({price_change_pct:+.2f}%)
• {get_label('trend_direction')}: {trend}

{get_label('predicted_range_section')}
• {get_label('maximum')}: ${predicted_max:,.2f}
• {get_label('minimum')}: ${predicted_min:,.2f}
• {get_label('price_range')}: ${predicted_max - predicted_min:,.2f}
• {get_label('predicted_volatility')}: ${predicted_volatility:,.2f}

{get_label('risk_assessment')}
• {get_label('volatility_ratio')}: {predicted_volatility / historical_volatility:.2f}x
• {get_label('risk_level')}: {get_label('high') if abs(price_change_pct) > 5 else get_label('medium') if abs(price_change_pct) > 2 else get_label('low')}
• {get_label('confidence')}: {get_label('high') if predicted_volatility < historical_volatility else get_label('medium')}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily=get_text_font(),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen'))
        ax.set_title(get_label('prediction_analysis'), fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def create_unified_time_axis(self, hist_df: pd.DataFrame, true_df: pd.DataFrame, pred_df: pd.DataFrame):
        """
        创建统一的时间轴，让预测数据和真实数据在图表上显示为相同长度
        
        Args:
            hist_df (pd.DataFrame): 历史数据
            true_df (pd.DataFrame): 真实数据（用于对比）
            pred_df (pd.DataFrame): 预测数据
            
        Returns:
            tuple: (hist_x, hist_y, true_x, true_y, pred_x, pred_y, x_labels, x_positions)
        """
        # 创建统一的X轴索引
        hist_len = len(hist_df)
        true_len = len(true_df)
        pred_len = len(pred_df)
        
        # 历史数据的X轴位置 (0 到 hist_len-1)
        hist_x = np.arange(hist_len)
        hist_y = hist_df['close'].values
        
        # 真实数据的X轴位置 (从 hist_len 开始)
        true_x = np.arange(hist_len, hist_len + true_len)
        true_y = true_df['close'].values
        
        # 预测数据的X轴位置 (也从 hist_len 开始，与真实数据重叠)
        pred_x = np.arange(hist_len, hist_len + pred_len)
        pred_y = pred_df['close'].values
        
        # 创建时间标签 - 选择关键时间点显示
        total_points = hist_len + max(true_len, pred_len)
        label_positions = []
        label_texts = []
        
        # 添加历史数据的起始和结束时间
        if hist_len > 0:
            label_positions.extend([0, hist_len - 1])
            label_texts.extend([
                hist_df['timestamps'].iloc[0].strftime('%m-%d %H:%M'),
                hist_df['timestamps'].iloc[-1].strftime('%m-%d %H:%M')
            ])
        
        # 添加预测数据的结束时间
        if pred_len > 0:
            label_positions.append(hist_len + pred_len - 1)
            if hasattr(pred_df.index, 'strftime'):
                label_texts.append(pred_df.index[-1].strftime('%m-%d %H:%M'))
            else:
                # 如果预测数据没有时间索引，使用真实数据的时间
                if true_len > 0:
                    label_texts.append(true_df['timestamps'].iloc[-1].strftime('%m-%d %H:%M'))
                else:
                    label_texts.append('预测终点')
        
        return hist_x, hist_y, true_x, true_y, pred_x, pred_y, label_texts, label_positions
    
    def create_unified_volume_axis(self, hist_df: pd.DataFrame, true_df: pd.DataFrame, pred_df: pd.DataFrame):
        """
        创建统一的成交量轴
        
        Args:
            hist_df (pd.DataFrame): 历史数据
            true_df (pd.DataFrame): 真实数据
            pred_df (pd.DataFrame): 预测数据
            
        Returns:
            tuple: (hist_x, hist_vol, true_x, true_vol, pred_x, pred_vol)
        """
        hist_len = len(hist_df)
        true_len = len(true_df)
        pred_len = len(pred_df)
        
        hist_x = np.arange(hist_len)
        hist_vol = hist_df['volume'].values
        
        true_x = np.arange(hist_len, hist_len + true_len)
        true_vol = true_df['volume'].values
        
        pred_x = np.arange(hist_len, hist_len + pred_len)
        pred_vol = pred_df['volume'].values
        
        return hist_x, hist_vol, true_x, true_vol, pred_x, pred_vol

    def add_performance_metrics(self, ax, true_df: pd.DataFrame, pred_df: pd.DataFrame):
        """添加性能指标到图表"""
        # 确保数据长度一致
        min_len = min(len(true_df), len(pred_df))
        if len(true_df) != len(pred_df):
            logging.warning(f"⚠️ 性能指标计算：数据长度不一致，true_df: {len(true_df)}, pred_df: {len(pred_df)}")
            logging.info(f"   使用前 {min_len} 个数据点进行计算")
            true_values = true_df['close'].values[:min_len]
            pred_values = pred_df['close'].values[:min_len]
        else:
            true_values = true_df['close'].values
            pred_values = pred_df['close'].values
        
        # 计算指标
        price_error = abs(true_values - pred_values)
        mae = price_error.mean()
        rmse = np.sqrt((price_error ** 2).mean())
        mape = (price_error / true_values * 100).mean()
        
        actual_changes = np.diff(true_values)
        predicted_changes = np.diff(pred_values)
        direction_accuracy = np.mean(np.sign(actual_changes) == np.sign(predicted_changes)) * 100
        
        actual_volatility = pd.Series(true_values).std()
        predicted_volatility = pd.Series(pred_values).std()
        volatility_ratio = predicted_volatility / actual_volatility
        
        actual_returns = pd.Series(true_values).pct_change().dropna()
        predicted_returns = pd.Series(pred_values).pct_change().dropna()
        correlation = actual_returns.corr(predicted_returns) if len(actual_returns) > 1 else 0
        
        # 性能指标文本
        stats_text = f"""{get_label('prediction_performance_metrics')}

{get_label('accuracy_measures')}
• {get_label('mae')}: ${mae:,.0f}
• {get_label('rmse')}: ${rmse:,.0f}
• {get_label('mape')}: {mape:.2f}%
• {get_label('direction_accuracy')}: {direction_accuracy:.1f}%
• {get_label('price_correlation')}: {correlation:.3f}

{get_label('price_range_analysis')}
• {get_label('actual_range')}: ${true_values.min():,.0f} - ${true_values.max():,.0f}
• {get_label('predicted_range')}: ${pred_values.min():,.0f} - ${pred_values.max():,.0f}

{get_label('volatility_analysis')}
• {get_label('actual_volatility')}: ${actual_volatility:,.0f}
• {get_label('predicted_volatility')}: ${predicted_volatility:,.0f}
• {get_label('volatility_capture')}: {volatility_ratio:.1%}

{get_label('model_assessment')}
• {get_label('overall_grade')}: {'A' if mape < 2 else 'B' if mape < 5 else 'C' if mape < 10 else 'D'}
• {get_label('trend_capture')}: {get_label('good') if direction_accuracy > 55 else get_label('fair') if direction_accuracy > 45 else get_label('poor')}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily=get_text_font(),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='navy'))
        ax.set_title(get_label('performance_analysis'), fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def predict_single(self, symbol: str = None) -> bool:
        """
        预测单个股票
        
        Args:
            symbol (str): 股票代码，如果为 None 则使用配置文件中的第一个
            
        Returns:
            bool: 是否成功
        """
        if symbol is None:
            symbol = self.config.get_single_symbol(0)
        
        # 下载数据
        data_file = self.download_data(symbol)
        if not data_file:
            return False
        
        # 进行预测
        result = self.predict_stock(symbol, data_file)
        return result is not None
    
    def predict_batch(self) -> List[Tuple[str, bool]]:
        """
        批量预测多个股票
        
        Returns:
            List[Tuple[str, bool]]: 每个股票的预测结果 (symbol, success)
        """
        batch_config = self.config.get_batch_config()
        
        if not batch_config.get('enabled', False):
            logging.warning("批量处理未启用")
            return []
        
        symbols = batch_config.get('batch_symbols') or self.config.get_symbols()
        max_workers = batch_config.get('max_workers', 2)
        
        results = []
        
        if max_workers > 1:
            # 并行处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.predict_single, symbol): symbol for symbol in symbols}
                
                for future in futures:
                    symbol = futures[future]
                    try:
                        success = future.result()
                        results.append((symbol, success))
                    except Exception as e:
                        logging.error(f"❌ 处理 {symbol} 时出错: {e}")
                        results.append((symbol, False))
        else:
            # 串行处理
            for symbol in symbols:
                success = self.predict_single(symbol)
                results.append((symbol, success))
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于配置文件的股票预测")
    parser.add_argument('--config', '-c', default='config.yaml', help='配置文件路径')
    parser.add_argument('--symbol', '-s', help='指定要预测的股票代码')
    parser.add_argument('--batch', '-b', action='store_true', help='批量处理模式')
    parser.add_argument('--list-symbols', '-l', action='store_true', help='列出配置中的所有股票代码')
    parser.add_argument('--adaptive', '-a', action='store_true', help='使用自适应采样参数（根据标的类型自动调整）')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config_manager = ConfigManager(args.config)
        
        if args.list_symbols:
            print("📋 配置文件中的股票代码:")
            for i, symbol in enumerate(config_manager.get_symbols()):
                print(f"  {i+1}. {symbol}")
            return
        
        # 显示配置
        config_manager.print_config()
        
        # 创建预测器
        predictor = StockPredictor(config_manager, use_adaptive=args.adaptive)
        
        if args.batch:
            # 批量预测
            logging.info("🚀 开始批量预测...")
            results = predictor.predict_batch()
            
            print("\n📊 批量预测结果:")
            for symbol, success in results:
                status = "✅ 成功" if success else "❌ 失败"
                print(f"  {symbol}: {status}")
        else:
            # 单个预测
            success = predictor.predict_single(args.symbol)
            if success:
                logging.info("🎉 预测完成！")
            else:
                logging.error("❌ 预测失败")
                
    except Exception as e:
        logging.error(f"❌ 程序执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
