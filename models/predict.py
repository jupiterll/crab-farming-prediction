import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("河蟹养殖水质智能预测系统 - 综合预测")
print("=" * 70)

model_dir = r'c:\Users\15683\PycharmProjects\Crab river\models'

print("\n[步骤1] 加载所有模型...")
try:
    lstm_model = load_model(os.path.join(model_dir, 'water_quality_lstm_model.keras'))
    risk_model = joblib.load(os.path.join(model_dir, 'risk_warning_rf_model.pkl'))
    feeding_model = joblib.load(os.path.join(model_dir, 'feeding_optimization_gb_model.pkl'))
    
    scaler_X_lstm = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
    scaler_y_lstm = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))
    feature_cols_lstm = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
    time_steps = joblib.load(os.path.join(model_dir, 'time_steps.pkl'))
    
    risk_scaler = joblib.load(os.path.join(model_dir, 'risk_scaler.pkl'))
    risk_label_encoder = joblib.load(os.path.join(model_dir, 'risk_label_encoder.pkl'))
    risk_feature_cols = joblib.load(os.path.join(model_dir, 'risk_feature_cols.pkl'))
    
    feeding_scaler_X = joblib.load(os.path.join(model_dir, 'feeding_scaler_X.pkl'))
    feeding_scaler_y = joblib.load(os.path.join(model_dir, 'feeding_scaler_y.pkl'))
    feeding_feature_cols = joblib.load(os.path.join(model_dir, 'feeding_feature_cols.pkl'))
    
    print("所有模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

print("\n[步骤2] 创建特征工程函数...")
def create_features_for_prediction(df):
    """为预测创建特征工程"""
    df = df.copy()
    
    for col in ['氨氮-常规', 'COD-常规', '活性磷-常规', '溶解氧']:
        if col in df.columns:
            for lag in [1, 2, 3, 7]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
            for window in [3, 7]:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
    
    if '饲料投喂量（kg/塘）' in df.columns:
        for lag in [1, 2, 3]:
            df[f'饲料投喂量_lag{lag}'] = df['饲料投喂量（kg/塘）'].shift(lag)
    
    if '肥料投喂量（kg/塘）' in df.columns:
        for lag in [1, 2, 3]:
            df[f'肥料投喂量_lag{lag}'] = df['肥料投喂量（kg/塘）'].shift(lag)
    
    df = df.bfill().ffill().fillna(0)
    
    return df

print("\n[步骤3] 创建综合预测函数...")
def comprehensive_prediction(current_data, recent_data):
    """
    综合预测函数
    
    参数:
    - current_data: 当前水质数据 (DataFrame, 单行)
    - recent_data: 最近多天的数据 (DataFrame, 用于特征工程)
    
    返回:
    - 完整的预测结果和建议
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("预测结果")
    print("=" * 70)
    
    print("\n【1. 水质预测】")
    try:
        recent_data_with_features = create_features_for_prediction(recent_data)
        
        recent_7days = recent_data_with_features.tail(time_steps)
        
        missing_cols = set(feature_cols_lstm) - set(recent_7days.columns)
        for col in missing_cols:
            recent_7days[col] = 0
        
        X_recent = recent_7days[feature_cols_lstm].values
        X_scaled = scaler_X_lstm.transform(X_recent)
        X_seq = X_scaled.reshape(1, time_steps, -1)
        
        y_pred_scaled = lstm_model.predict(X_seq, verbose=0)
        y_pred = scaler_y_lstm.inverse_transform(y_pred_scaled)
        
        print(f"  预测明天的水质指标:")
        print(f"    - 氨氮:   {y_pred[0, 0]:.2f} mg/L")
        print(f"    - COD:    {y_pred[0, 1]:.2f} mg/L")
        print(f"    - 活性磷: {y_pred[0, 2]:.2f} mg/L")
        print(f"    - 溶解氧: {y_pred[0, 3]:.2f} mg/L")
        
        results['water_quality_prediction'] = {
            '氨氮': float(y_pred[0, 0]),
            'COD': float(y_pred[0, 1]),
            '活性磷': float(y_pred[0, 2]),
            '溶解氧': float(y_pred[0, 3])
        }
    except Exception as e:
        print(f"  水质预测失败: {e}")
        results['water_quality_prediction'] = None
    
    print("\n【2. 风险预警】")
    try:
        X_risk = current_data[risk_feature_cols].values
        X_risk_scaled = risk_scaler.transform(X_risk)
        
        risk_pred = risk_model.predict(X_risk_scaled)
        risk_prob = risk_model.predict_proba(X_risk_scaled)
        
        risk_level = risk_label_encoder.inverse_transform(risk_pred)[0]
        risk_prob_dict = dict(zip(risk_label_encoder.classes_, risk_prob[0]))
        
        print(f"  当前风险等级: {risk_level}")
        print(f"  各等级概率:")
        for level, prob in risk_prob_dict.items():
            print(f"    - {level}: {prob*100:.1f}%")
        
        results['risk_warning'] = {
            'risk_level': risk_level,
            'probabilities': {k: float(v) for k, v in risk_prob_dict.items()}
        }
    except Exception as e:
        print(f"  风险预警失败: {e}")
        results['risk_warning'] = None
    
    print("\n【3. 投料优化建议】")
    try:
        X_feed = current_data[feeding_feature_cols].values
        X_feed_scaled = feeding_scaler_X.transform(X_feed)
        
        y_feed_scaled = feeding_model.predict(X_feed_scaled)
        y_feed = feeding_scaler_y.inverse_transform(y_feed_scaled)
        
        optimal_feed = y_feed[0, 0]
        optimal_fertilizer = y_feed[0, 1]
        
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
        
        print(f"  推荐饲料投喂量: {optimal_feed:.1f} kg/塘")
        print(f"  推荐肥料投喂量: {optimal_fertilizer:.1f} kg/塘")
        print(f"  投喂建议:")
        for rec in recommendations:
            print(f"    - {rec}")
        
        results['feeding_optimization'] = {
            'optimal_feed': float(optimal_feed),
            'optimal_fertilizer': float(optimal_fertilizer),
            'recommendations': recommendations
        }
    except Exception as e:
        print(f"  投料优化失败: {e}")
        results['feeding_optimization'] = None
    
    return results

print("\n[步骤4] 示例预测...")
print("\n使用示例数据演示预测功能...")

data_path = r'c:\Users\15683\PycharmProjects\Crab river\data\optimized_data.csv'
df = pd.read_csv(data_path)

df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values(['日期', '池塘类型\n（Pond)']).reset_index(drop=True)

treatment_data = df[df['池塘类型\n（Pond)'] == '环沟推水'].copy()

recent_data = treatment_data.tail(20)
current_data = treatment_data.tail(1)

print(f"\n当前数据日期: {current_data['日期'].values[0]}")
print(f"当前水质状况:")
print(f"  - 氨氮:   {current_data['氨氮-常规'].values[0]:.2f} mg/L")
print(f"  - COD:    {current_data['COD-常规'].values[0]:.2f} mg/L")
print(f"  - 活性磷: {current_data['活性磷-常规'].values[0]:.2f} mg/L")
print(f"  - 溶解氧: {current_data['溶解氧'].values[0]:.2f} mg/L")
print(f"  - 水温:   {current_data['池塘温度'].values[0]:.1f} ℃")

results = comprehensive_prediction(current_data, recent_data)

print("\n" + "=" * 70)
print("预测完成!")
print("=" * 70)

print("\n[步骤5] 保存预测函数...")
def predict_for_website(current_data_dict, recent_data_list):
    """
    为网站提供的预测接口
    
    参数:
    - current_data_dict: 当前水质数据字典
    - recent_data_list: 最近多天数据列表
    
    返回:
    - JSON格式的预测结果
    """
    current_df = pd.DataFrame([current_data_dict])
    recent_df = pd.DataFrame(recent_data_list)
    
    results = comprehensive_prediction(current_df, recent_df)
    
    return results

print("\n网站预测接口已创建")

print("\n" + "=" * 70)
print("使用说明")
print("=" * 70)
print("""
1. 水质预测模型:
   - 输入: 最近多天的水质数据
   - 输出: 预测明天的氨氮、COD、活性磷、溶解氧

2. 风险预警模型:
   - 输入: 当前水质数据
   - 输出: 风险等级（低风险/中风险/高风险）及概率

3. 投料优化模型:
   - 输入: 当前水质数据
   - 输出: 推荐的饲料和肥料投喂量及建议

4. 网站集成:
   - 使用 predict_for_website() 函数
   - 输入字典格式的数据
   - 输出JSON格式的预测结果
""")
