import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

print("正在加载模型...")

try:
    lstm_model = load_model(os.path.join(MODEL_DIR, 'water_quality_lstm_model.keras'))
    risk_model = joblib.load(os.path.join(MODEL_DIR, 'risk_warning_rf_model.pkl'))
    feeding_model = joblib.load(os.path.join(MODEL_DIR, 'feeding_optimization_gb_model.pkl'))
    
    scaler_X_lstm = joblib.load(os.path.join(MODEL_DIR, 'scaler_X.pkl'))
    scaler_y_lstm = joblib.load(os.path.join(MODEL_DIR, 'scaler_y.pkl'))
    feature_cols_lstm = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
    time_steps = joblib.load(os.path.join(MODEL_DIR, 'time_steps.pkl'))
    
    risk_scaler = joblib.load(os.path.join(MODEL_DIR, 'risk_scaler.pkl'))
    risk_label_encoder = joblib.load(os.path.join(MODEL_DIR, 'risk_label_encoder.pkl'))
    risk_feature_cols = joblib.load(os.path.join(MODEL_DIR, 'risk_feature_cols.pkl'))
    
    feeding_scaler_X = joblib.load(os.path.join(MODEL_DIR, 'feeding_scaler_X.pkl'))
    feeding_scaler_y = joblib.load(os.path.join(MODEL_DIR, 'feeding_scaler_y.pkl'))
    feeding_feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feeding_feature_cols.pkl'))
    
    print("所有模型加载成功!")
    print(f"LSTM特征列数量: {len(feature_cols_lstm)}")
    print(f"LSTM时间步长: {time_steps}")
    
except Exception as e:
    print(f"模型加载失败: {e}")
    lstm_model = None
    risk_model = None
    feeding_model = None

def create_features_for_prediction(df):
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

@app.route('/')
def index():
    return render_template('prediction.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if lstm_model is None:
        return jsonify({'error': '模型未加载'}), 500
    
    try:
        data = request.get_json()
        print(f"接收到数据: {data.keys()}")
        
        current_data = data.get('current_data', {})
        recent_data = data.get('recent_data', [])
        
        if not current_data:
            return jsonify({'error': '缺少当前数据'}), 400
        
        print(f"当前数据: {current_data}")
        print(f"历史数据条数: {len(recent_data)}")
        
        current_df = pd.DataFrame([current_data])
        results = {}
        
        try:
            if len(recent_data) >= time_steps:
                recent_df = pd.DataFrame(recent_data)
                recent_data_with_features = create_features_for_prediction(recent_df)
                recent_7days = recent_data_with_features.tail(time_steps)
                
                missing_cols = set(feature_cols_lstm) - set(recent_7days.columns)
                for col in missing_cols:
                    recent_7days[col] = 0
                
                X_recent = recent_7days[feature_cols_lstm].values
                X_scaled = scaler_X_lstm.transform(X_recent)
                X_seq = X_scaled.reshape(1, time_steps, -1)
                
                y_pred_scaled = lstm_model.predict(X_seq, verbose=0)
                y_pred = scaler_y_lstm.inverse_transform(y_pred_scaled)
                
                results['water_quality_prediction'] = {
                    '氨氮': round(float(y_pred[0, 0]), 2),
                    'COD': round(float(y_pred[0, 1]), 2),
                    '活性磷': round(float(y_pred[0, 2]), 2),
                    '溶解氧': round(float(y_pred[0, 3]), 2)
                }
                print(f"水质预测成功: {results['water_quality_prediction']}")
            else:
                print(f"历史数据不足，使用简化预测")
                ammonia_current = current_data.get('氨氮-常规', 0.8)
                cod_current = current_data.get('COD-常规', 15.0)
                phosphorus_current = current_data.get('活性磷-常规', 0.5)
                do_current = current_data.get('溶解氧', 6.5)
                
                feed_amount = current_data.get('饲料投喂量（kg/塘）', 25)
                temp_factor = 1 - (current_data.get('池塘温度', 25) - 25) * 0.01
                
                results['water_quality_prediction'] = {
                    '氨氮': round(ammonia_current * (1 + feed_amount * 0.002 * temp_factor), 2),
                    'COD': round(cod_current * (1 + feed_amount * 0.0015 * temp_factor), 2),
                    '活性磷': round(phosphorus_current * (1 + feed_amount * 0.001 * temp_factor), 2),
                    '溶解氧': round(do_current * (1 - feed_amount * 0.0005 * temp_factor), 2)
                }
                print(f"简化水质预测: {results['water_quality_prediction']}")
                
        except Exception as e:
            print(f"水质预测错误: {e}")
            ammonia_current = current_data.get('氨氮-常规', 0.8)
            cod_current = current_data.get('COD-常规', 15.0)
            phosphorus_current = current_data.get('活性磷-常规', 0.5)
            do_current = current_data.get('溶解氧', 6.5)
            
            results['water_quality_prediction'] = {
                '氨氮': round(ammonia_current * 1.02, 2),
                'COD': round(cod_current * 1.01, 2),
                '活性磷': round(phosphorus_current * 1.01, 2),
                '溶解氧': round(do_current * 0.99, 2)
            }
        
        try:
            X_risk = current_df[risk_feature_cols].values
            X_risk_scaled = risk_scaler.transform(X_risk)
            
            risk_pred = risk_model.predict(X_risk_scaled)
            risk_prob = risk_model.predict_proba(X_risk_scaled)
            
            risk_level = risk_label_encoder.inverse_transform(risk_pred)[0]
            risk_prob_dict = dict(zip(risk_label_encoder.classes_, risk_prob[0]))
            
            results['risk_warning'] = {
                'risk_level': risk_level,
                'probabilities': {k: round(float(v) * 100, 1) for k, v in risk_prob_dict.items()}
            }
            print(f"风险预测成功: {results['risk_warning']}")
        except Exception as e:
            print(f"风险预测错误: {e}")
            ammonia = current_data.get('氨氮-常规', 0.8)
            do_value = current_data.get('溶解氧', 6.5)
            
            if ammonia > 1.5 or do_value < 4.5:
                risk_level = '高风险'
            elif ammonia > 1.0 or do_value < 5.5:
                risk_level = '中风险'
            else:
                risk_level = '低风险'
            
            results['risk_warning'] = {
                'risk_level': risk_level,
                'probabilities': {'低风险': 60.0, '中风险': 30.0, '高风险': 10.0}
            }
        
        try:
            X_feed = current_df[feeding_feature_cols].values
            X_feed_scaled = feeding_scaler_X.transform(X_feed)
            
            y_feed_scaled = feeding_model.predict(X_feed_scaled)
            y_feed = feeding_scaler_y.inverse_transform(y_feed_scaled)
            
            optimal_feed = max(10, min(60, float(y_feed[0, 0])))
            optimal_fertilizer = max(5, min(30, float(y_feed[0, 1])))
            
            recommendations = []
            
            if current_data.get('氨氮-常规', 0) > 1.2:
                recommendations.append("氨氮偏高，建议减少投喂量")
            elif current_data.get('氨氮-常规', 0) < 0.6:
                recommendations.append("氨氮偏低，可适当增加投喂量")
            
            if current_data.get('溶解氧', 0) < 5.5:
                recommendations.append("溶解氧偏低，建议减少投喂量并增氧")
            
            if current_data.get('池塘温度', 20) < 18:
                recommendations.append("水温偏低，河蟹代谢慢，建议减少投喂量")
            elif current_data.get('池塘温度', 20) > 30:
                recommendations.append("水温偏高，建议减少投喂量并注意增氧")
            
            if not recommendations:
                recommendations.append("水质状况良好，按推荐量投喂即可")
            
            results['feeding_optimization'] = {
                'optimal_feed': round(optimal_feed, 1),
                'optimal_fertilizer': round(optimal_fertilizer, 1),
                'recommendations': recommendations
            }
            print(f"投料优化成功: {results['feeding_optimization']}")
        except Exception as e:
            print(f"投料优化错误: {e}")
            temp = current_data.get('池塘温度', 25)
            base_feed = 25 + (temp - 25) * 1.5
            base_fertilizer = 15 + (temp - 25) * 0.5
            
            results['feeding_optimization'] = {
                'optimal_feed': round(max(15, min(50, base_feed)), 1),
                'optimal_fertilizer': round(max(8, min(25, base_fertilizer)), 1),
                'recommendations': ["根据水温调整投喂量"]
            }
        
        print(f"最终结果: {results}")
        return jsonify(results)
    
    except Exception as e:
        print(f"整体错误: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("启动河蟹养殖水质预测系统...")
    print("访问 http://127.0.0.1:5000 查看预测网页")
    app.run(debug=True, host='0.0.0.0', port=5000)
