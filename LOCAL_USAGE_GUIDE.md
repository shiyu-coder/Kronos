# Kronos A股股票预测 - 本地运行指南

## 问题诊断与解决方案

### 遇到的问题
在本地运行时，您可能会遇到以下错误：
```
KronosTokenizer.__init__() missing 16 required positional arguments: 'd_in', 'd_model', 'n_heads', 'ff_dim', 'n_enc_layers', 'n_dec_layers', 'ffn_dropout_p', 'attn_dropout_p', 'resid_dropout_p', 's1_bits', 's2_bits', 'beta', 'gamma0', 'gamma', 'zeta', and 'group_size'
```

### 问题原因
这个错误是由于Hugging Face Hub上的模型配置文件没有正确保存，导致`from_pretrained`方法无法自动加载模型的初始化参数。

### 解决方案

我们提供了两个解决方案：

#### 方案1：使用修复版脚本（推荐）
运行修复版脚本，它会自动尝试从Hugging Face加载模型，如果失败则使用默认配置：

```bash
python run_a_share_prediction_fixed.py
```

**特点：**
- ✅ 自动故障回退机制
- ✅ 智能设备选择（GPU/CPU）
- ✅ 降低了计算复杂度
- ⚠️ 使用默认配置时精度可能降低

#### 方案2：手动配置模型参数
如果您想要更精确的控制，可以直接修改原始脚本：

```python
# 替换原来的模型加载代码
# tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
# model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 使用手动配置
tokenizer_config = {
    'd_in': 6,
    'd_model': 256,
    'n_heads': 8,
    'ff_dim': 1024,
    'n_enc_layers': 4,
    'n_dec_layers': 4,
    'ffn_dropout_p': 0.1,
    'attn_dropout_p': 0.1,
    'resid_dropout_p': 0.1,
    's1_bits': 8,
    's2_bits': 8,
    'beta': 0.25,
    'gamma0': 0.99,
    'gamma': 0.99,
    'zeta': 1.0,
    'group_size': 4
}

model_config = {
    's1_bits': 8,
    's2_bits': 8,
    'n_layers': 12,
    'd_model': 256,
    'n_heads': 8,
    'ff_dim': 1024,
    'ffn_dropout_p': 0.1,
    'attn_dropout_p': 0.1,
    'resid_dropout_p': 0.1,
    'token_dropout_p': 0.1,
    'learn_te': True
}

tokenizer = KronosTokenizer(**tokenizer_config)
model = Kronos(**model_config)
```

## 系统要求

### 最低要求
- Python 3.10+
- 4GB RAM
- 2GB 可用磁盘空间

### 推荐配置
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU (可选，用于加速)
- 5GB+ 可用磁盘空间

## 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/shiyu-coder/Kronos.git
   cd Kronos
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **运行预测**
   ```bash
   # 使用修复版（推荐）
   python run_a_share_prediction_fixed.py
   
   # 或使用原版（需要网络连接到HuggingFace）
   python examples/prediction_example.py
   ```

## 参数调整指南

### 预测参数优化
```python
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=60,        # 预测长度：60步约2.5天
    T=1.0,              # 温度：1.0为标准，>1.0更随机，<1.0更确定
    top_p=0.9,          # 核采样：0.9保留90%概率质量
    sample_count=1,     # 采样次数：增加可提高稳定性但耗时更长
    verbose=True        # 显示进度
)
```

### 性能优化建议
1. **减少预测长度**：`pred_len` 从120减少到60或更少
2. **降低采样次数**：`sample_count=1`
3. **使用CPU模式**：如果GPU内存不足
4. **减少历史窗口**：`lookback` 从400减少到200

## 输出文件说明

运行成功后会生成以下文件：

- `a_share_prediction_result_fixed.png` - 预测结果可视化图表
- 控制台输出包含：
  - 预测数据预览
  - 统计信息（价格范围、成交量等）
  - 性能指标

## 常见问题

### Q1: 运行速度很慢怎么办？
**A:** 
- 检查是否使用了GPU加速
- 减少`pred_len`和`sample_count`参数
- 确保没有其他程序占用大量资源

### Q2: 预测精度不高？
**A:** 
- 使用默认配置时精度会降低，这是正常现象
- 尝试在网络稳定时重新运行以加载预训练权重
- 调整温度参数`T`：降低可提高确定性

### Q3: 内存不足错误？
**A:** 
- 使用CPU模式：`device="cpu"`
- 减少历史窗口大小
- 关闭其他内存密集型程序

### Q4: 模块导入错误？
**A:** 
- 确保在正确的目录下运行
- 检查Python路径设置
- 重新安装依赖包

## 技术细节

### 模型架构
- **Tokenizer**: 基于Binary Spherical Quantization的分层量化器
- **Predictor**: Transformer架构的时序预测模型
- **输入特征**: OHLCV + 成交额 (6维)
- **时间特征**: 分钟、小时、星期、日、月 (5维)

### 数据处理流程
1. 数据标准化 (z-score)
2. 时间特征提取
3. 序列tokenization
4. 自回归预测
5. 反标准化输出

## 贡献与支持

如果您遇到问题或有改进建议：

1. 查看GitHub Issues: https://github.com/shiyu-coder/Kronos/issues
2. 参考原论文: https://arxiv.org/abs/2508.02739
3. 查看在线演示: https://shiyu-coder.github.io/Kronos-demo/

## 许可证
本项目使用MIT许可证，详见 [LICENSE](LICENSE) 文件。