# ========== Step 0: 基础导入 ==========
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ========== Step 1: 加载数据并重命名列 ==========
@st.cache_data
def load_data():
    data1 = pd.read_csv("PFP-zixuan-筛选.csv", encoding='gbk')
    data1.dropna(inplace=True)

    data1.columns = [
        'Number',
        'Target variable',
        'Ankle flexion/extension angle',
        'Knee ad/abduction angle',
        'Pelvic obliquity angle',
        'Pelvic rotation angle',
        'Lumbar rotation angle',
        'M/L COM position',
        'Vertical COM position',
        'Biceps femoris short head',
        'Adductor brevis',
        'Tensor fasciae latae',
        'Piriformis',
        'Rectus femoris',
        'Vastus lateralis',
        'Medial gastrocnemius',
        'Soleus',
    ]
    return data1

data = load_data()

# ========== Step 2: 特征与标签 ==========
features = data.columns[2:]  # 从第3列开始为特征
X = data[features]
y = data['Target variable']

# ========== Step 3: 拆分训练集与测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== Step 4: 模型训练 ==========
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=1,
    min_child_weight=3,
    learning_rate=0.04,
    n_estimators=200,
    subsample=0.7,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# ========== Step 5: Streamlit 前端界面 ==========
st.title("Patellofemoral Joint Load Prediction Tool")

st.write("""
Based on the movement parameters you entered on the left,  
the model has estimated your **patellofemoral joint load** during this running session as shown below.  
These results are derived from our model, which was developed to explore the relationship among **running kinematics, muscle activation patterns, and patellofemoral joint loading** in recreational runners with irregular running habits.  
Please note that the predicted values represent **model-based estimations**, which may exhibit minor deviations from actual measurements.
""")

# 用户输入
st.sidebar.header("Input Parameters")
input_values = []
for col in features:
    val = st.sidebar.slider(col, -180.0, 180.0, 0.0)
    input_values.append(val)

user_input_df = pd.DataFrame([input_values], columns=features)

# ========== Step 6: 预测结果展示 ==========
predicted_load = model.predict(user_input_df)

st.subheader("Prediction Result")
st.write(f"**Predicted Patellofemoral Joint Load:** `{predicted_load[0]:.2f}` N/kg")


