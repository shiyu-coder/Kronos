#!/usr/bin/env python3
"""
最终中文字体修复方案
适用于各种操作系统和环境
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os
import platform
import requests
import tempfile
from pathlib import Path

def download_chinese_font():
    """下载中文字体文件"""
    print("📥 正在下载中文字体...")
    try:
        # 使用思源黑体，开源且支持中文
        font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
        
        # 创建字体目录
        font_dir = Path.home() / ".matplotlib" / "fonts"
        font_dir.mkdir(parents=True, exist_ok=True)
        
        font_path = font_dir / "SourceHanSansSC-Regular.otf"
        
        if not font_path.exists():
            print("   - 下载思源黑体字体文件...")
            response = requests.get(font_url, timeout=30)
            response.raise_for_status()
            
            with open(font_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✅ 字体已下载到: {font_path}")
        else:
            print(f"   ✅ 字体文件已存在: {font_path}")
        
        return str(font_path)
    except Exception as e:
        print(f"   ❌ 字体下载失败: {e}")
        return None

def setup_chinese_font_robust():
    """健壮的中文字体设置方案"""
    print("🔧 设置matplotlib中文字体...")
    
    # 方案1：尝试系统已安装的中文字体
    system_fonts = [
        'SimHei',           # 黑体 (Windows)
        'Microsoft YaHei',  # 微软雅黑 (Windows)
        'PingFang SC',      # Mac
        'Hiragino Sans GB', # Mac
        'WenQuanYi Micro Hei', # Linux
        'Noto Sans CJK SC', # Google Noto (跨平台)
        'Source Han Sans SC', # 思源黑体
    ]
    
    available_fonts = []
    for font_name in fm.findSystemFonts():
        try:
            font_prop = fm.FontProperties(fname=font_name)
            if font_prop.get_name() in system_fonts:
                available_fonts.append(font_prop.get_name())
        except:
            continue
    
    if available_fonts:
        chosen_font = available_fonts[0]
        matplotlib.rcParams['font.sans-serif'] = [chosen_font] + matplotlib.rcParams['font.sans-serif']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print(f"   ✅ 使用系统字体: {chosen_font}")
        return chosen_font
    
    # 方案2：下载并使用开源中文字体
    print("   - 系统中未找到中文字体，尝试下载...")
    downloaded_font = download_chinese_font()
    if downloaded_font:
        try:
            # 清除字体缓存
            fm._rebuild()
            
            # 注册字体
            fm.fontManager.addfont(downloaded_font)
            
            # 设置字体
            font_prop = fm.FontProperties(fname=downloaded_font)
            font_name = font_prop.get_name()
            matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            print(f"   ✅ 使用下载字体: {font_name}")
            return font_name
        except Exception as e:
            print(f"   ❌ 字体注册失败: {e}")
    
    # 方案3：使用英文替代并提示用户
    print("   ⚠️  无法设置中文字体，使用英文标签")
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False
    return "English Fallback"

def create_chinese_friendly_plot():
    """创建对中文友好的图表"""
    font_name = setup_chinese_font_robust()
    
    # 测试字体效果
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 如果有中文字体，使用中文标签
    if font_name != "English Fallback":
        title = "A股股票预测结果 - 中文字体测试"
        xlabel = "时间"
        ylabel = "价格"
        legend_labels = ["真实价格", "预测价格"]
    else:
        title = "A-Share Stock Prediction - Font Test"
        xlabel = "Time"
        ylabel = "Price"
        legend_labels = ["Actual Price", "Predicted Price"]
    
    # 创建示例数据
    import numpy as np
    x = np.linspace(0, 100, 100)
    y1 = 10 + 0.5 * np.sin(x * 0.1) + np.random.normal(0, 0.1, 100)
    y2 = 10 + 0.3 * np.sin(x * 0.1 + 0.5) + np.random.normal(0, 0.1, 100)
    
    ax.plot(x, y1, label=legend_labels[0], color='blue', linewidth=2)
    ax.plot(x, y2, label=legend_labels[1], color='red', linewidth=2, linestyle='--')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chinese_font_test_final.png', dpi=300, bbox_inches='tight')
    
    print(f"📊 测试图表已保存为: chinese_font_test_final.png")
    print(f"🔤 当前字体: {font_name}")
    
    return font_name

def main():
    """主函数"""
    print("🚀 开始最终中文字体修复测试")
    print("=" * 50)
    
    try:
        font_name = create_chinese_friendly_plot()
        
        print("\n✅ 字体设置完成!")
        print(f"📋 使用字体: {font_name}")
        
        if font_name == "English Fallback":
            print("\n💡 建议:")
            print("   - 在Windows上安装SimHei或微软雅黑字体")
            print("   - 在Mac上使用PingFang SC字体")
            print("   - 在Linux上安装 'sudo apt-get install fonts-wqy-microhei'")
            print("   - 或运行 pip install matplotlib[fonts] 获取更多字体支持")
        else:
            print("\n🎉 中文字体设置成功!")
            print("   现在可以正常显示中文了")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()