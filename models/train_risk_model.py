import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("河蟹养殖风险预警模型训练")
print("=" * 60)

data_path = r'c:\Users\15683\PycharmProjects\Crab river\data\optimized_data.csv'
model_dir = r'c:\Users\15683\PycharmProjects\Crab river\models'
os.makedirs(model_dir, exist_ok=True)

print("\n[步骤1] 加载数据...")
df = pd.read_csv(data_path)
print(f"数据形状: {df.shape}")

print("\n[步骤2] 定义风险等级...")
def calculate_risk_level(row):
    risk_score = 0
    
    if row['氨氮-常规'] > 1.5:
        risk_score += 2
    elif row['氨氮-常规'] > 1.0:
        risk_score += 1
    
    if row['COD-常规'] > 20:
        risk_score += 2
    elif row['COD-常规'] > 15:
        risk_score += 1
    
    if row['活性磷-常规'] > 1.2:
        risk_score += 2
    elif row['活性磷-常规'] > 0.8:
        risk_score += 1
    
    if row['溶解氧'] < 5:
        risk_score += 2
    elif row['溶解氧'] < 6:
        risk_score += 1
    
    if row['池塘温度'] < 18 or row['池塘温度'] > 30:
        risk_score += 1
    
    if risk_score <= 2:
        return '低风险'
    elif risk_score <= 5:
        return '中风险'
    else:
        return '高风险'

df['风险等级'] = df.apply(calculate_risk_level, axis=1)
print(f"风险等级分布:\n{df['风险等级'].value_counts()}")

print("\n[步骤3] 准备特征...")
feature_cols = [
    '氨氮-常规', 'COD-常规', '活性磷-常规', '溶解氧',
    '气温（T)', '池塘温度',
    '饲料投喂量（kg/塘）', '肥料投喂量（kg/塘）',
    '池塘类型编码', '月份', '星期', '季节'
]

available_features = [col for col in feature_cols if col in df.columns]
print(f"可用特征: {available_features}")

X = df[available_features].copy()
y = df['风险等级'].copy()

X = X.dropna()
y = y[X.index]

print(f"特征形状: {X.shape}")
print(f"标签形状: {y.shape}")

print("\n[步骤4] 数据标准化和编码...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"风险等级编码: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

print("\n[步骤5] 划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

print("\n[步骤6] 训练随机森林模型...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

print("\n[步骤7] 训练梯度提升模型...")
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)

print("\n[步骤8] 模型评估...")
print("\n随机森林模型评估:")
rf_pred = rf_model.predict(X_test)
print(f"准确率: {accuracy_score(y_test, rf_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

print("\n梯度提升模型评估:")
gb_pred = gb_model.predict(X_test)
print(f"准确率: {accuracy_score(y_test, gb_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, gb_pred, target_names=label_encoder.classes_))

print("\n[步骤9] 保存模型...")
joblib.dump(rf_model, os.path.join(model_dir, 'risk_warning_rf_model.pkl'))
joblib.dump(gb_model, os.path.join(model_dir, 'risk_warning_gb_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'risk_scaler.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'risk_label_encoder.pkl'))
joblib.dump(available_features, os.path.join(model_dir, 'risk_feature_cols.pkl'))

print("\n模型已保存到:", model_dir)

print("\n[步骤10] 创建预警函数...")
def predict_risk_level(current_data, model, scaler, label_encoder, feature_cols):
    """
    预测风险等级
    
    参数:
    - current_data: 当前水质数据 (DataFrame)
    - model: 训练好的模型
    - scaler: 数据标准化器
    - label_encoder: 标签编码器
    - feature_cols: 特征列名
    
    返回:
    - 风险等级: '低风险', '中风险', '高风险'
    - 各风险等级的概率
    """
    X = current_data[feature_cols].values
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)
    
    risk_level = label_encoder.inverse_transform(prediction)[0]
    risk_prob = dict(zip(label_encoder.classes_, probability[0]))
    
    return risk_level, risk_prob

print("\n预警函数已创建")

print("\n" + "=" * 60)
print("风险预警模型训练完成!")
print("=" * 60)
print("\n保存的文件:")
print("1. risk_warning_rf_model.pkl - 随机森林模型")
print("2. risk_warning_gb_model.pkl - 梯度提升模型")
print("3. risk_scaler.pkl - 特征标准化器")
print("4. risk_label_encoder.pkl - 标签编码器")
print("5. risk_feature_cols.pkl - 特征列名")
print("\n使用方法:")
print("1. 加载模型和预处理器")
print("2. 准备当前水质数据")
print("3. 调用predict_risk_level函数进行风险预测")
