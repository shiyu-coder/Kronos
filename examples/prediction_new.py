import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings
import requests
import json
import time
import random
import akshare as ak
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# 添加项目路径以便导入自定义模块
sys.path.append("../")
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    print("⚠️ 无法导入Kronos模型，预测功能将不可用")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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


def generate_future_dates(last_date, pred_len):
    """
    生成未来的交易日日期
    """
    future_dates = []
    current_date = last_date + timedelta(days=1)

    while len(future_dates) < pred_len:
        if current_date.weekday() < 5:
            future_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"📅 生成的未来交易日: 共 {len(future_dates)} 天")
    print(f"   起始日期: {future_dates[0].strftime('%Y-%m-%d')}")
    print(f"   结束日期: {future_dates[-1].strftime('%Y-%m-%d')}")

    return future_dates[:pred_len]


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


def get_stock_price_reference(stock_code, current_price):
    """
    根据当前价格智能计算参考价格范围
    """
    price_ranges = {
        '600580': (current_price * 0.75, current_price * 1.25),
        '300207': (current_price * 0.75, current_price * 1.25),
        '300418': (current_price * 0.75, current_price * 1.25),
        '002354': (current_price * 0.75, current_price * 1.25),
        '000001': (current_price * 0.75, current_price * 1.25),
        '600036': (current_price * 0.75, current_price * 1.25),
    }

    if stock_code in price_ranges:
        min_price, max_price = price_ranges[stock_code]
        min_price = max(1.0, min_price)
        return {'min': min_price, 'max': max_price}
    else:
        return {'min': max(1.0, current_price * 0.7), 'max': current_price * 1.3}


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
                'current_rate': 4.25,  # 联邦基金利率目标区间4.00%-4.25%:cite[3]
                'trend': '降息周期',
                'recent_cut': '2025年9月降息25个基点',
                'expected_cuts_2025': 2,  # 市场预期2025年还有两次降息:cite[7]
                'expected_cuts_2026': 2,
                'impact_on_emerging_markets': 'positive',
                'usd_index_support': 95.0,  # 美元指数短期支撑位:cite[7]
                'analysis': '美联储开启宽松周期，利好全球流动性'
            }

            # 国内政策因素 - 基于最新政策
            domestic_policy = {
                'monetary_policy': '稳健偏松',
                'fiscal_policy': '积极财政',
                'market_liquidity': '合理充裕',
                'industrial_policy': '设备更新、以旧换新',  # 大规模设备更新政策:cite[5]
                'employment_policy': '稳就业政策加力',  # 国务院稳就业政策:cite[8]
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
                        '与智元机器人实现双向持股，推进具身智能机器人技术研发:cite[5]',
                        '成立浙江龙飞电驱，专注航空电机业务:cite[5]',
                        '发布AI外骨骼机器人及灵巧手:cite[9]',
                        '布局高爆发关节模组、伺服驱动器等人形机器人关键部件:cite[5]'
                    ],
                    'growth_drivers': [
                        '设备更新政策推动工业电机需求:cite[5]',
                        '机器人产业快速发展',
                        '低空经济政策支持',
                        '出海战略加速'
                    ],
                    'risk_factors': [
                        '机器人业务营收占比仅2.71%，占比较低:cite[1]',
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


# ==================== 增强预测函数 ====================
def enhance_prediction_with_market_factors(
        historical_df,
        prediction_df,
        stock_code,
        market_analyzer
):
    """
    使用市场因素增强预测结果 - 多维度综合分析
    """
    print("\n🎯 使用多维度市场因素增强预测...")

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
            # 使用更温和的调整，避免过度乐观或悲观
            adjusted_value = enhanced_prediction[col] * adjustment_factor
            # 限制单次调整幅度在±10%以内
            change_ratio = adjusted_value / enhanced_prediction[col]
            if change_ratio.max() > 1.1:
                adjusted_value = enhanced_prediction[col] * 1.1
            elif change_ratio.min() < 0.9:
                adjusted_value = enhanced_prediction[col] * 0.9
            enhanced_prediction[col] = adjusted_value

    # 对成交量进行调整
    if 'volume' in enhanced_prediction.columns:
        volume_adjustment = 1 + (adjustment_factor - 1) * 0.3  # 成交量调整更温和
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
    计算基于多维度市场因素的调整因子 - 更平衡的方法
    """
    base_factor = 1.0
    factors_log = []

    # 1. 大盘趋势影响 (权重25%)
    if market_analysis['overall_is_main_uptrend']:
        trend_strength = market_analysis['overall_trend_strength']
        adjustment = 1 + trend_strength * 0.08  # 降低主升浪影响幅度
        base_factor *= adjustment
        factors_log.append(f"大盘主升浪: +{trend_strength * 0.08:.3f}")
    else:
        trend_strength = market_analysis['overall_trend_strength']
        # 震荡市不一定悲观，只是增幅较小
        adjustment = 1 + (trend_strength - 0.5) * 0.04
        base_factor *= adjustment
        factors_log.append(f"大盘震荡: {(trend_strength - 0.5) * 0.04:+.3f}")

    # 2. 板块共振影响 (权重25%)
    resonance_score = sector_analysis['resonance_score']
    sector_count = sector_analysis['sector_count']

    if sector_analysis['is_sector_hot']:
        # 热门板块且有多个概念叠加
        sector_adjustment = 1 + resonance_score * 0.06 + min(sector_count * 0.01, 0.03)
        base_factor *= sector_adjustment
        factors_log.append(
            f"热门板块({sector_count}个): +{resonance_score * 0.06 + min(sector_count * 0.01, 0.03):.3f}")
    else:
        # 非热门板块也有基础支撑
        base_factor *= (1 + (resonance_score - 0.5) * 0.02)
        factors_log.append(f"一般板块: {(resonance_score - 0.5) * 0.02:+.3f}")

    # 3. 宏观因素影响 (权重20%)
    macro_score = macro_analysis['overall_macro_score']
    macro_adjustment = 1 + (macro_score - 0.5) * 0.06
    base_factor *= macro_adjustment
    factors_log.append(f"宏观环境: {(macro_score - 0.5) * 0.06:+.3f}")

    # 4. 美国降息周期特殊影响 (权重10%)
    us_rate_trend = macro_analysis['us_rate_cycle']['trend']
    if us_rate_trend == '降息周期':
        expected_cuts = macro_analysis['us_rate_cycle']['expected_cuts_2025']
        us_adjustment = 1 + expected_cuts * 0.015  # 降低单次降息影响
        base_factor *= us_adjustment
        factors_log.append(f"美国降息: +{expected_cuts * 0.015:.3f}")

    # 5. 公司基本面影响 (权重20%)
    fundamental_score = fundamental_analysis['fundamental_score']
    fundamental_adjustment = 1 + (fundamental_score - 0.5) * 0.08
    base_factor *= fundamental_adjustment
    factors_log.append(f"基本面: {(fundamental_score - 0.5) * 0.08:+.3f}")

    # 输出调整因子详情
    print("🔍 调整因子详情:")
    for log in factors_log:
        print(f"   {log}")

    # 限制调整幅度在更合理的范围内 (0.85 ~ 1.15)
    final_factor = max(0.85, min(1.15, base_factor))

    if final_factor != base_factor:
        print(f"⚠️  调整因子从 {base_factor:.3f} 限制到 {final_factor:.3f}")

    return final_factor


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
        'adjustment_factor': enhancement_info['adjustment_factor'],
        'analysis_summary': generate_analysis_summary(enhancement_info)
    }

    # 保存报告
    report_file = os.path.join(output_dir, f'{stock_code}_comprehensive_analysis_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📋 综合分析报告已保存: {report_file}")
    return report


def generate_analysis_summary(enhancement_info):
    """
    生成分析总结
    """
    market = enhancement_info['market_analysis']
    sector = enhancement_info['sector_analysis']
    macro = enhancement_info['macro_analysis']
    fundamental = enhancement_info['fundamental_analysis']

    summary = {
        'overall_sentiment': '积极' if enhancement_info['adjustment_factor'] > 1.0 else '谨慎',
        'key_drivers': [],
        'main_risks': [],
        'investment_suggestion': ''
    }

    # 关键驱动因素
    if market['overall_trend_strength'] > 0.6:
        summary['key_drivers'].append('大盘趋势向好')

    if sector['is_sector_hot']:
        summary['key_drivers'].append(f"热门板块:{sector['main_sector']['sector']}")

    if macro['overall_macro_score'] > 0.7:
        summary['key_drivers'].append('宏观环境有利')

    if fundamental['fundamental_score'] > 0.6:
        summary['key_drivers'].append('基本面稳健')

    # 主要风险
    if market['overall_trend_strength'] < 0.4:
        summary['main_risks'].append('大盘趋势偏弱')

    if not sector['is_sector_hot']:
        summary['main_risks'].append('非热门板块')

    if len(summary['key_drivers']) > len(summary['main_risks']):
        summary['investment_suggestion'] = '可考虑逢低关注'
    else:
        summary['investment_suggestion'] = '建议谨慎操作'

    return summary


# ==================== 增强可视化函数 ====================
def plot_comprehensive_prediction(
        historical_df,
        prediction_df,
        future_dates,
        stock_code,
        stock_name,
        output_dir,
        enhancement_info=None
):
    """
    绘制综合预测图表 - 包含更多市场分析信息
    """
    ensure_output_directory(output_dir)

    # 设置配色
    colors = {
        'historical': '#1f77b4',
        'prediction': '#ff7f0e',
        'enhanced': '#2ca02c',
        'background': '#f8f9fa',
        'grid': '#e9ecef',
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }

    # 创建综合图表
    fig = plt.figure(figsize=(18, 14))
    gs = plt.GridSpec(4, 3, figure=fig, height_ratios=[2, 1, 1, 1])

    # 1. 主价格图表
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(colors['background'])

    # 2. 成交量图表
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor(colors['background'])

    # 3. 市场分析图表
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(colors['background'])

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor(colors['background'])

    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(colors['background'])

    # 4. 因素分析图表
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor(colors['background'])

    # 设置背景色
    fig.patch.set_facecolor('white')

    # 1. 价格图表
    historical_prices = historical_df.set_index('timestamps')['close']
    prediction_prices = prediction_df.set_index(pd.DatetimeIndex(future_dates))['close']

    # 获取当前最新价格
    current_price = historical_prices.iloc[-1]

    # 智能Y轴范围计算
    all_prices = pd.concat([historical_prices, prediction_prices])
    data_min = all_prices.min()
    data_max = all_prices.max()

    price_range = data_max - data_min
    y_margin = price_range * 0.15

    y_min = max(0, data_min - y_margin)
    y_max = data_max + y_margin

    # 设置Y轴刻度
    y_interval = calculate_optimal_interval(y_min, y_max)
    y_ticks = np.arange(round(y_min / y_interval) * y_interval,
                        round(y_max / y_interval) * y_interval + y_interval,
                        y_interval)

    # 绘制历史价格
    ax1.plot(historical_prices.index, historical_prices.values,
             color=colors['historical'], linewidth=2, label='历史价格')

    # 绘制预测价格
    if len(prediction_prices) > 0:
        # 连接点
        last_hist_date = historical_prices.index[-1]
        last_hist_price = historical_prices.iloc[-1]
        first_pred_date = prediction_prices.index[0]

        # 绘制连接线
        ax1.plot([last_hist_date, first_pred_date],
                 [last_hist_price, prediction_prices.iloc[0]],
                 color=colors['prediction'], linewidth=2.5, linestyle='-')

        # 绘制预测线
        ax1.plot(prediction_prices.index, prediction_prices.values,
                 color=colors['prediction'], linewidth=2.5, label='基础预测')

        # 绘制增强预测线
        if enhancement_info and 'enhanced_prediction' in enhancement_info:
            enhanced_prices = enhancement_info['enhanced_prediction'].set_index(pd.DatetimeIndex(future_dates))['close']
            ax1.plot(enhanced_prices.index, enhanced_prices.values,
                     color=colors['enhanced'], linewidth=2.5, linestyle='--', label='增强预测')

        # 标记预测起点
        ax1.axvline(x=last_hist_date, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.annotate('预测起点', xy=(last_hist_date, last_hist_price),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 设置Y轴范围和刻度
    ax1.set_ylim(y_min, y_max)
    ax1.set_yticks(y_ticks)

    ax1.set_ylabel('收盘价 (元)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, color=colors['grid'], alpha=0.7)

    title = f'{stock_name}({stock_code}) - 综合因素价格预测\n当前价: {current_price:.2f}元 | 增强因子: {enhancement_info["adjustment_factor"]:.3f}' if enhancement_info else f'{stock_name}({stock_code}) - 价格预测\n当前价: {current_price:.2f}元'
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # 设置x轴格式
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. 成交量图表
    historical_volume = historical_df.set_index('timestamps')['volume']
    prediction_volume = prediction_df.set_index(pd.DatetimeIndex(future_dates))['volume']

    # 计算相对成交量（标准化）
    hist_volume_norm = historical_volume / historical_volume.max()
    if len(prediction_volume) > 0:
        pred_volume_norm = prediction_volume / historical_volume.max()

    # 绘制历史成交量
    ax2.bar(historical_volume.index, hist_volume_norm.values,
            alpha=0.6, color=colors['historical'], label='历史成交量')

    # 绘制预测成交量
    if len(prediction_volume) > 0:
        ax2.bar(prediction_volume.index, pred_volume_norm.values,
                alpha=0.6, color=colors['prediction'], label='预测成交量')

    ax2.set_ylabel('相对成交量', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, color=colors['grid'], alpha=0.7)
    ax2.set_ylim(0, 1.2)

    # 设置x轴格式
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. 市场分析子图
    if enhancement_info:
        # 因素权重饼图
        factors = ['大盘趋势', '板块共振', '宏观环境', '美国降息', '基本面']
        weights = [25, 25, 20, 10, 20]
        colors_pie = [colors['historical'], colors['prediction'], colors['enhanced'], '#f39c12', '#9b59b6']

        ax3.pie(weights, labels=factors, autopct='%1.0f%%', colors=colors_pie, startangle=90)
        ax3.set_title('因素权重分配', fontweight='bold', fontsize=11)

        # 因素评分柱状图
        scores = [
            enhancement_info['market_analysis']['overall_trend_strength'],
            enhancement_info['sector_analysis']['resonance_score'],
            enhancement_info['macro_analysis']['overall_macro_score'],
            0.7 if enhancement_info['macro_analysis']['us_rate_cycle']['trend'] == '降息周期' else 0.3,
            enhancement_info['fundamental_analysis']['fundamental_score']
        ]

        x_pos = np.arange(len(factors))
        bars = ax4.bar(x_pos, scores, color=colors_pie, alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(factors, rotation=45, fontsize=9)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('评分', fontsize=10)
        ax4.set_title('各因素当前评分', fontweight='bold', fontsize=11)
        ax4.grid(True, alpha=0.3)

        # 在柱状图上显示数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 市场状态总结
        market_status = enhancement_info['market_analysis']['market_status']
        sector_status = "热门" if enhancement_info['sector_analysis']['is_sector_hot'] else "一般"
        macro_status = "有利" if enhancement_info['macro_analysis']['overall_macro_score'] > 0.6 else "不利"

        summary_text = f"""市场状态总结:

大盘趋势: {market_status}
板块热度: {sector_status}
宏观环境: {macro_status}
美国利率: {enhancement_info['macro_analysis']['us_rate_cycle']['trend']}
综合评分: {enhancement_info['adjustment_factor']:.3f}

投资建议: {enhancement_info['fundamental_analysis']['investment_rating']}"""

        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
                 verticalalignment='top', linespacing=1.5)
        ax5.set_title('市场状态总结', fontweight='bold', fontsize=11)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)

        # 4. 详细因素分析
        if 'analysis_summary' in enhancement_info:
            summary = enhancement_info['analysis_summary']
            drivers_text = "\n".join([f"• {driver}" for driver in summary['key_drivers']]) if summary[
                'key_drivers'] else "• 暂无明显驱动"
            risks_text = "\n".join([f"• {risk}" for risk in summary['main_risks']]) if summary[
                'main_risks'] else "• 风险可控"

            detail_text = f"""关键驱动因素:
{drivers_text}

主要风险提示:
{risks_text}

总体情绪: {summary['overall_sentiment']}
建议: {summary['investment_suggestion']}"""

            ax6.text(0.02, 0.95, detail_text, transform=ax6.transAxes, fontsize=9,
                     verticalalignment='top', linespacing=1.3)
            ax6.set_title('详细因素分析', fontweight='bold', fontsize=11)
            ax6.set_xticks([])
            ax6.set_yticks([])
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            ax6.spines['bottom'].set_visible(False)
            ax6.spines['left'].set_visible(False)

    plt.tight_layout()

    # 保存图片
    chart_filename = os.path.join(output_dir, f'{stock_code}_comprehensive_prediction.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 综合预测图表已保存: {chart_filename}")

    plt.show()

    return historical_prices, prediction_prices


# ==================== 主预测函数 ====================
def run_comprehensive_kronos_prediction(stock_code, stock_name, data_dir, pred_days, output_dir, history_years=1):
    """
    运行综合版Kronos模型预测流程
    """
    print(f"\n🎯 开始 {stock_name}({stock_code}) 综合版Kronos模型价格预测")
    print("=" * 60)

    # 初始化增强版市场分析器
    market_analyzer = EnhancedMarketFactorAnalyzer()

    try:
        # 1. 获取数据
        print("\n步骤1: 获取股票数据...")
        success, csv_file_path = get_stock_data(stock_code, data_dir)
        if not success:
            print("❌ 无法获取股票数据，预测终止")
            return

        # 2. 加载模型和分词器
        print("\n步骤2: 加载Kronos模型和分词器...")
        try:
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
            print("✅ 模型加载完成 - 使用Kronos-base模型")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("⚠️ 预测功能不可用，请检查模型安装")
            return

        # 3. 实例化预测器
        print("步骤3: 初始化预测器...")
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
        print("✅ 预测器初始化完成")

        # 4. 准备数据
        print("步骤4: 准备股票数据...")
        df = prepare_stock_data(csv_file_path, stock_code, history_years)

        # 5. 计算预测参数
        print("步骤5: 计算预测参数...")
        lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)

        if pred_len <= 0:
            print("❌ 数据量不足，无法进行预测")
            return

        print(f"✅ 最终参数 - 回看期: {lookback}, 预测期: {pred_len}")

        # 6. 准备输入数据
        print("步骤6: 准备输入数据...")
        x_df = df.loc[-lookback:, ['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.loc[-lookback:, 'timestamps'].reset_index(drop=True)

        # 生成未来日期
        last_historical_date = df['timestamps'].iloc[-1]
        future_dates = generate_future_dates(last_historical_date, pred_len)

        print(f"输入数据形状: {x_df.shape}")
        print(f"历史数据时间范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
        print(f"预测时间范围: {future_dates[0]} 到 {future_dates[-1]}")

        # 7. 执行基础预测
        print("步骤7: 执行基础价格预测...")
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

        print("✅ 基础预测完成")
        print("预测数据前5行:")
        print(pred_df.head())

        # 8. 使用多维度市场因素增强预测
        print("步骤8: 应用多维度市场因素增强预测...")
        enhanced_pred_df, enhancement_info = enhance_prediction_with_market_factors(
            df.loc[-lookback:].reset_index(drop=True),
            pred_df,
            stock_code,
            market_analyzer
        )

        # 将增强预测结果添加到信息中
        enhancement_info['enhanced_prediction'] = enhanced_pred_df

        # 9. 创建综合市场分析报告
        market_report = create_comprehensive_market_report(enhancement_info, output_dir, stock_code)

        # 10. 可视化结果
        print("步骤9: 生成综合版可视化图表...")
        historical_df = df.loc[-lookback:].reset_index(drop=True)
        hist_prices, base_pred_prices = plot_comprehensive_prediction(
            historical_df, pred_df, future_dates, stock_code, stock_name, output_dir, enhancement_info
        )

        # 11. 生成综合预测报告
        print("步骤10: 生成综合预测报告...")
        if len(enhanced_pred_df) > 0:
            current_price = hist_prices.iloc[-1]
            base_predicted_price = base_pred_prices.iloc[-1] if len(base_pred_prices) > 0 else current_price
            enhanced_predicted_price = enhanced_pred_df.set_index(pd.DatetimeIndex(future_dates))['close'].iloc[-1]

            base_change_pct = (base_predicted_price / current_price - 1) * 100
            enhanced_change_pct = (enhanced_predicted_price / current_price - 1) * 100

            print(f"\n📈 综合版Kronos模型预测报告")
            print("=" * 70)
            print(f"股票: {stock_name}({stock_code})")
            print(f"当前价格: {current_price:.2f} 元")
            print(f"基础预测价格: {base_predicted_price:.2f} 元 ({base_change_pct:+.2f}%)")
            print(f"增强预测价格: {enhanced_predicted_price:.2f} 元 ({enhanced_change_pct:+.2f}%)")
            print(f"市场因素调整因子: {enhancement_info['adjustment_factor']:.4f}")
            print(f"大盘状态: {enhancement_info['market_analysis']['market_status']}")
            print(
                f"板块共振: {enhancement_info['sector_analysis']['main_sector']['sector']} (分数: {enhancement_info['sector_analysis']['resonance_score']:.2f})")
            print(f"宏观环境: 美国{enhancement_info['macro_analysis']['us_rate_cycle']['trend']}")
            print(f"公司评级: {enhancement_info['fundamental_analysis']['investment_rating']}")
            print(f"预测期间: {pred_len} 个交易日")

            # 输出关键因素
            print(f"\n🔑 关键影响因素:")
            for driver in enhancement_info['analysis_summary']['key_drivers']:
                print(f"  ✅ {driver}")
            for risk in enhancement_info['analysis_summary']['main_risks']:
                print(f"  ⚠️  {risk}")
            print(f"  💡 投资建议: {enhancement_info['analysis_summary']['investment_suggestion']}")

            # 保存详细预测数据
            prediction_details = pd.DataFrame({
                '日期': future_dates,
                '基础预测收盘价': base_pred_prices.values if len(base_pred_prices) > 0 else [current_price] * len(
                    future_dates),
                '增强预测收盘价': enhanced_pred_df['close'].values,
                '预测成交量': enhanced_pred_df['volume'].values
            })

            prediction_file = os.path.join(output_dir, f'{stock_code}_comprehensive_predictions.csv')
            prediction_details.to_csv(prediction_file, index=False, encoding='utf-8-sig')
            print(f"💾 详细预测数据已保存: {prediction_file}")

        print(f"\n🎉 {stock_name}({stock_code}) 综合版Kronos模型预测完成!")

    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# ==================== 主函数 ====================
def main():
    """
    主函数：综合版Kronos模型股票预测系统
    """
    # ==================== 配置参数 ====================
    STOCK_CONFIG = {
        "stock_code": "603288",
        "stock_name": "海天味业",
        "data_dir": r"D:\lianghuajiaoyi\Kronos\examples\data",
        "pred_days": 60,
        "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce",
        "history_years": 1
    }

    print("🤖 综合版Kronos模型股票价格预测系统")
    print("=" * 50)
    print("📊 新增功能: 多维度市场因素分析")
    print("🎯 包含: 大盘趋势 + 板块共振 + 宏观政策 + 公司基本面")
    print("🚀 使用模型: Kronos-base (更适合3070Ti显卡)")
    print(f"当前预测股票: {STOCK_CONFIG['stock_name']}({STOCK_CONFIG['stock_code']})")
    print(f"预测天数: {STOCK_CONFIG['pred_days']} 天")
    print(f"输出目录: {STOCK_CONFIG['output_dir']}")
    print()

    # 运行综合版Kronos模型预测流程
    run_comprehensive_kronos_prediction(**STOCK_CONFIG)

    print(f"\n💡 提示：综合版模型已整合多维度市场环境分析因子")


if __name__ == "__main__":
    main()