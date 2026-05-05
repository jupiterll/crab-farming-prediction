import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("河蟹养殖水质智能预测系统 - 模型训练")
print("=" * 60)

data_path = r'c:\Users\15683\PycharmProjects\Crab river\data\optimized_data.csv'
model_dir = r'c:\Users\15683\PycharmProjects\Crab river\models'
os.makedirs(model_dir, exist_ok=True)

print("\n[步骤1] 加载数据...")
df = pd.read_csv(data_path)
print(f"数据形状: {df.shape}")
print(f"数据列: {list(df.columns)}")

print("\n[步骤2] 数据预处理...")
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values(['日期', '池塘类型\n（Pond)']).reset_index(drop=True)

feature_cols = [
    '氨氮-常规', 'COD-常规', '活性磷-常规', '溶解氧',
    '气温（T)', '池塘温度',
    '饲料投喂量（kg/塘）', '肥料投喂量（kg/塘）',
    '池塘类型编码', '月份', '星期', '季节'
]

available_features = [col for col in feature_cols if col in df.columns]
print(f"可用特征: {available_features}")

df_clean = df[['日期', '池塘类型\n（Pond)'] + available_features].copy()
df_clean = df_clean.dropna()
print(f"清洗后数据量: {len(df_clean)}")

print("\n[步骤3] 特征工程...")
def create_features(df):
    df = df.copy()
    
    for col in ['氨氮-常规', 'COD-常规', '活性磷-常规', '溶解氧']:
        if col in df.columns:
            for lag in [1, 2, 3, 7]:
                df[f'{col}_lag{lag}'] = df.groupby('池塘类型\n（Pond)')[col].shift(lag)
            
            for window in [3, 7]:
                df[f'{col}_rolling_mean_{window}'] = df.groupby('池塘类型\n（Pond)')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_{window}'] = df.groupby('池塘类型\n（Pond)')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
    
    if '饲料投喂量（kg/塘）' in df.columns:
        for lag in [1, 2, 3]:
            df[f'饲料投喂量_lag{lag}'] = df.groupby('池塘类型\n（Pond)')['饲料投喂量（kg/塘）'].shift(lag)
    
    if '肥料投喂量（kg/塘）' in df.columns:
        for lag in [1, 2, 3]:
            df[f'肥料投喂量_lag{lag}'] = df.groupby('池塘类型\n（Pond)')['肥料投喂量（kg/塘）'].shift(lag)
    
    return df

df_features = create_features(df_clean)
df_features = df_features.dropna()
print(f"特征工程后数据量: {len(df_features)}")

print("\n[步骤4] 准备训练数据...")
target_cols = ['氨氮-常规', 'COD-常规', '活性磷-常规', '溶解氧']
feature_cols_final = [col for col in df_features.columns 
                      if col not in ['日期', '池塘类型\n（Pond)'] + target_cols]

print(f"特征数量: {len(feature_cols_final)}")
print(f"目标变量: {target_cols}")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = df_features[feature_cols_final].values
y = df_features[target_cols].values

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

print(f"X形状: {X_scaled.shape}")
print(f"y形状: {y_scaled.shape}")

def create_sequences(X, y, time_steps=7):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 7
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
print(f"序列形状: X={X_seq.shape}, y={y_seq.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

print("\n[步骤5] 构建LSTM模型...")
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(time_steps, X_seq.shape[2]))),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(4)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

print("\n[步骤6] 训练模型...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(model_dir, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\n[步骤7] 模型评估...")
y_pred = model.predict(X_test)
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

print("\n各指标评估结果:")
metrics_results = {}
for i, col in enumerate(target_cols):
    mse = mean_squared_error(y_test_original[:, i], y_pred_original[:, i])
    mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
    
    metrics_results[col] = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    print(f"\n{col}:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")

print("\n[步骤8] 保存模型和预处理器...")
model.save(os.path.join(model_dir, 'water_quality_lstm_model.keras'))
joblib.dump(scaler_X, os.path.join(model_dir, 'scaler_X.pkl'))
joblib.dump(scaler_y, os.path.join(model_dir, 'scaler_y.pkl'))
joblib.dump(feature_cols_final, os.path.join(model_dir, 'feature_cols.pkl'))
joblib.dump(target_cols, os.path.join(model_dir, 'target_cols.pkl'))
joblib.dump(time_steps, os.path.join(model_dir, 'time_steps.pkl'))

print("\n模型和预处理器已保存到:", model_dir)

print("\n[步骤9] 创建预测函数...")
def predict_water_quality(recent_data, model, scaler_X, scaler_y, feature_cols, time_steps):
    """
    预测未来水质
    
    参数:
    - recent_data: 最近time_steps天的数据 (DataFrame)
    - model: 训练好的模型
    - scaler_X, scaler_y: 数据标准化器
    - feature_cols: 特征列名
    - time_steps: 时间步长
    
    返回:
    - 预测的水质指标: [氨氮, COD, 活性磷, 溶解氧]
    """
    X = recent_data[feature_cols].values
    X_scaled = scaler_X.transform(X)
    X_seq = X_scaled.reshape(1, time_steps, -1)
    
    y_pred_scaled = model.predict(X_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return y_pred[0]

print("\n预测函数已创建")

print("\n" + "=" * 60)
print("模型训练完成!")
print("=" * 60)
print("\n保存的文件:")
print("1. water_quality_lstm_model.keras - LSTM模型")
print("2. scaler_X.pkl - 特征标准化器")
print("3. scaler_y.pkl - 目标变量标准化器")
print("4. feature_cols.pkl - 特征列名")
print("5. target_cols.pkl - 目标列名")
print("6. time_steps.pkl - 时间步长")
print("\n使用方法:")
print("1. 加载模型和预处理器")
print("2. 准备最近7天的数据")
print("3. 调用predict_water_quality函数进行预测")
