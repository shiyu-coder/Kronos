#!/usr/bin/env python3
"""
测试matplotlib中文字体显示
用于验证中文乱码修复是否生效
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def setup_chinese_font():
    """设置matplotlib中文字体"""
    try:
        # 尝试常见的中文字体
        chinese_fonts = [
            'SimHei',           # 黑体 (Windows)
            'Microsoft YaHei',  # 微软雅黑 (Windows)
            'DejaVu Sans',      # Linux通用字体
            'Arial Unicode MS', # Mac
            'PingFang SC',      # Mac
            'Hiragino Sans GB', # Mac
            'WenQuanYi Micro Hei', # Linux
            'sans-serif'        # 默认
        ]
        
        used_font = None
        # 逐个尝试字体
        for font in chinese_fonts:
            try:
                matplotlib.rcParams['font.family'] = font
                matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
                # 测试是否支持中文
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '测试', fontsize=12)
                plt.close(fig)
                print(f"✅ 成功设置中文字体: {font}")
                used_font = font
                break
            except:
                continue
        else:
            print("⚠️  未找到合适的中文字体，使用默认字体")
            used_font = "默认字体"
        
        # 设置负号正常显示
        matplotlib.rcParams['axes.unicode_minus'] = False
        return used_font
        
    except Exception as e:
        print(f"⚠️  字体设置失败: {e}")
        # 使用备用方案：直接指定sans-serif
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['axes.unicode_minus'] = False
        return "备用字体"

def test_chinese_display():
    """测试中文显示效果"""
    print("🧪 开始测试matplotlib中文字体显示")
    print("=" * 50)
    
    # 设置中文字体
    used_font = setup_chinese_font()
    
    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上图：简单的中文测试
    ax1.plot(x, y1, label='正弦波 (sin)', color='blue', linewidth=2)
    ax1.plot(x, y2, label='余弦波 (cos)', color='red', linewidth=2)
    ax1.set_title('中文字体显示测试 - 数学函数图', fontsize=16, fontweight='bold')
    ax1.set_xlabel('横轴 (X轴)', fontsize=12)
    ax1.set_ylabel('纵轴 (Y轴)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 下图：A股相关术语测试
    stock_terms = ['开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额']
    values = [10.5, 11.2, 10.3, 10.8, 15000, 160000]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    bars = ax2.bar(stock_terms, values, color=colors, alpha=0.8)
    ax2.set_title('A股股票数据可视化测试', fontsize=16, fontweight='bold')
    ax2.set_ylabel('数值', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,.0f}' if value > 1000 else f'{value:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('chinese_font_test.png', dpi=300, bbox_inches='tight')
    
    print(f"📊 测试图表已生成并保存为: chinese_font_test.png")
    print(f"🔤 当前使用字体: {used_font}")
    print("📋 测试内容包括:")
    print("   - 中文标题和标签")
    print("   - A股相关术语")
    print("   - 数学符号和负号")
    print("   - 各种字体大小")
    
    # 显示图表
    plt.show()
    
    # 输出字体信息
    print("\n📋 系统字体信息:")
    print(f"   当前字体家族: {matplotlib.rcParams['font.family']}")
    print(f"   Sans-serif字体: {matplotlib.rcParams['font.sans-serif'][:3]}")
    print(f"   负号显示设置: {not matplotlib.rcParams['axes.unicode_minus']}")
    
    return True

def main():
    """主函数"""
    try:
        success = test_chinese_display()
        if success:
            print("\n✅ 中文字体显示测试完成!")
            print("💡 如果图表中的中文显示正常，说明乱码问题已解决。")
            print("💡 如果仍有乱码，可能需要安装中文字体包。")
        else:
            print("\n❌ 测试失败!")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()