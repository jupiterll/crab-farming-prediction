import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("河蟹养殖投料优化建议模型训练")
print("=" * 60)

data_path = r'c:\Users\15683\PycharmProjects\Crab river\data\optimized_data.csv'
model_dir = r'c:\Users\15683\PycharmProjects\Crab river\models'
os.makedirs(model_dir, exist_ok=True)

print("\n[步骤1] 加载数据...")
df = pd.read_csv(data_path)
print(f"数据形状: {df.shape}")

print("\n[步骤2] 定义优化目标...")
def calculate_optimal_feeding(row):
    base_feed = 25
    base_fertilizer = 18
    
    if row['氨氮-常规'] > 1.2:
        base_feed -= 5
        base_fertilizer -= 3
    elif row['氨氮-常规'] < 0.6:
        base_feed += 3
    
    if row['COD-常规'] > 18:
        base_feed -= 4
        base_fertilizer -= 2
    
    if row['活性磷-常规'] > 1.0:
        base_fertilizer -= 4
    
    if row['溶解氧'] < 5.5:
        base_feed -= 6
        base_fertilizer -= 3
    elif row['溶解氧'] > 7.5:
        base_feed += 2
    
    if 20 <= row['池塘温度'] <= 28:
        base_feed += 5
        base_fertilizer += 2
    elif row['池塘温度'] < 18 or row['池塘温度'] > 30:
        base_feed -= 8
        base_fertilizer -= 4
    
    if row['月份'] in [6, 7, 8]:
        base_feed += 3
        base_fertilizer += 2
    elif row['月份'] in [3, 4, 10, 11]:
        base_feed -= 2
        base_fertilizer -= 1
    
    optimal_feed = max(15, min(40, base_feed))
    optimal_fertilizer = max(10, min(35, base_fertilizer))
    
    return optimal_feed, optimal_fertilizer

df[['最优饲料量', '最优肥料量']] = df.apply(
    lambda row: pd.Series(calculate_optimal_feeding(row)), axis=1
)

print(f"最优饲料量范围: {df['最优饲料量'].min():.1f} - {df['最优饲料量'].max():.1f} kg/塘")
print(f"最优肥料量范围: {df['最优肥料量'].min():.1f} - {df['最优肥料量'].max():.1f} kg/塘")

print("\n[步骤3] 准备特征...")
feature_cols = [
    '氨氮-常规', 'COD-常规', '活性磷-常规', '溶解氧',
    '气温（T)', '池塘温度',
    '池塘类型编码', '月份', '星期', '季节'
]

available_features = [col for col in feature_cols if col in df.columns]
print(f"可用特征: {available_features}")

X = df[available_features].copy()
y = df[['最优饲料量', '最优肥料量']].copy()

X = X.dropna()
y = y.loc[X.index]

print(f"特征形状: {X.shape}")
print(f"目标形状: {y.shape}")

print("\n[步骤4] 数据标准化...")
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

print("\n[步骤5] 划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

print("\n[步骤6] 训练随机森林模型...")
rf_model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
)
rf_model.fit(X_train, y_train)

print("\n[步骤7] 训练梯度提升模型...")
gb_model = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )
)
gb_model.fit(X_train, y_train)

print("\n[步骤8] 模型评估...")
def evaluate_model(model, X_test, y_test, scaler_y, model_name):
    y_pred = model.predict(X_test)
    
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    print(f"\n{model_name}评估结果:")
    
    for i, col in enumerate(['最优饲料量', '最优肥料量']):
        mse = mean_squared_error(y_test_original[:, i], y_pred_original[:, i])
        mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
        
        print(f"\n{col}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
    
    return y_pred_original, y_test_original

rf_pred, rf_test = evaluate_model(rf_model, X_test, y_test, scaler_y, "随机森林")
gb_pred, gb_test = evaluate_model(gb_model, X_test, y_test, scaler_y, "梯度提升")

print("\n[步骤9] 保存模型...")
joblib.dump(rf_model, os.path.join(model_dir, 'feeding_optimization_rf_model.pkl'))
joblib.dump(gb_model, os.path.join(model_dir, 'feeding_optimization_gb_model.pkl'))
joblib.dump(scaler_X, os.path.join(model_dir, 'feeding_scaler_X.pkl'))
joblib.dump(scaler_y, os.path.join(model_dir, 'feeding_scaler_y.pkl'))
joblib.dump(available_features, os.path.join(model_dir, 'feeding_feature_cols.pkl'))

print("\n模型已保存到:", model_dir)

print("\n[步骤10] 创建投料优化函数...")
def optimize_feeding(current_data, model, scaler_X, scaler_y, feature_cols):
    """
    优化投喂量建议
    
    参数:
    - current_data: 当前水质数据 (DataFrame)
    - model: 训练好的模型
    - scaler_X, scaler_y: 数据标准化器
    - feature_cols: 特征列名
    
    返回:
    - 最优饲料投喂量 (kg/塘)
    - 最优肥料投喂量 (kg/塘)
    - 投喂建议
    """
    X = current_data[feature_cols].values
    X_scaled = scaler_X.transform(X)
    
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    optimal_feed = y_pred[0, 0]
    optimal_fertilizer = y_pred[0, 1]
    
    recommendations = []
    
    if current_data['氨氮-常规'].values[0] > 1.2:
        recommendations.append("氨氮偏高，建议减少投喂量")
    elif current_data['氨氮-常规'].values[0] < 0.6:
        recommendations.append("氨氮偏低，可适当增加投喂量")
    
    if current_data['溶解氧'].values[0] < 5.5:
        recommendations.append("溶解氧偏低，建议减少投喂量并增氧")
    
    if current_data['池塘温度'].values[0] < 18:
        recommendations.append("水温偏低，河蟹代谢慢，建议减少投喂量")
    elif current_data['池塘温度'].values[0] > 30:
        recommendations.append("水温偏高，建议减少投喂量并注意增氧")
    
    if not recommendations:
        recommendations.append("水质状况良好，按推荐量投喂即可")
    
    return optimal_feed, optimal_fertilizer, recommendations

print("\n投料优化函数已创建")

print("\n" + "=" * 60)
print("投料优化建议模型训练完成!")
print("=" * 60)
print("\n保存的文件:")
print("1. feeding_optimization_rf_model.pkl - 随机森林模型")
print("2. feeding_optimization_gb_model.pkl - 梯度提升模型")
print("3. feeding_scaler_X.pkl - 特征标准化器")
print("4. feeding_scaler_y.pkl - 目标变量标准化器")
print("5. feeding_feature_cols.pkl - 特征列名")
print("\n使用方法:")
print("1. 加载模型和预处理器")
print("2. 准备当前水质数据")
print("3. 调用optimize_feeding函数获取投喂建议")
