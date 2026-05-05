# 河蟹养殖水质智能预测与可视化系统

## 项目简介
基于时序深度学习的蟹塘水质智能预测与可视化系统，用于河蟹养殖水质调控。系统采用LSTM模型，可提前12-24小时预警水质风险，为养殖户提供智能化决策支持。

## 项目结构

```
Crab river/
├── data/                      # 数据文件夹
│   ├── 河蟹数据总.xlsm       # 原始数据
│   ├── cleaned_data.csv      # 清洗后的数据
│   ├── improved_data.csv     # 改进后的数据
│   └── optimized_data.csv    # 优化后的数据（最终版本）
├── scripts/                   # 脚本文件夹
│   ├── comprehensive_data_processing.py  # 综合数据处理脚本
│   ├── model_training.py     # 模型训练脚本
│   └── visualization_system.py  # 可视化系统脚本
├── docs/                      # 文档文件夹
│   ├── 智农降水-耕区慧测平台25年.doc
│   ├── 气象智图-全景AI气象数据可视化与分析平台25年.doc
│   └── 2417408-林雅乐-专业基础能力实践大作业报告.doc
├── output/                    # 输出文件夹
└── README.md                  # 项目说明文档
```

## 使用说明

### 1. 数据处理
```bash
cd scripts
python comprehensive_data_processing.py
```
该脚本会自动完成数据清洗、改进和优化，生成最终的数据文件。

### 2. 模型训练
```bash
cd scripts
python model_training.py
```
该脚本会训练LSTM模型并生成预测结果。

### 3. 可视化展示
```bash
cd scripts
python visualization_system.py
```
该脚本会生成可视化网页，打开index.html即可查看系统总览。

## 技术栈
- **数据处理**: Python, Pandas, NumPy
- **模型训练**: TensorFlow, LSTM
- **可视化**: Plotly, HTML
- **部署**: 本地网页应用

## 数据说明
- **数据时间范围**: 2025年3月至11月（259天）
- **数据量**: 688条记录
- **预测指标**: 氨氮-常规, COD-常规, 活性磷-常规

## 项目特色
1. 基于LSTM的时间序列预测
2. 多源数据融合（水质、环境、管理）
3. 小样本优化算法
4. 轻量化部署方案
5. 直观的可视化界面


