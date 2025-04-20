
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# สร้างข้อมูลจำลองและฝึกโมเดล
# ---------------------------
np.random.seed(42)
rounds = 1000
results = np.random.choice(['P', 'B', 'T'], size=rounds, p=[0.45, 0.45, 0.10])

result_map = {'P': 0, 'B': 1, 'T': 2}
df = pd.DataFrame({'result': results})
df['result_code'] = df['result'].map(result_map)

for i in range(1, 4):
    df[f'prev_{i}'] = df['result_code'].shift(i)
df.dropna(inplace=True)

X = df[['prev_1', 'prev_2', 'prev_3']]
y = df['result_code']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Baccarat Outcome Predictor")
st.write("กรอกผลลัพธ์ย้อนหลัง 3 ตา เพื่อทำนายผลลัพธ์ตาถัดไป")

option_map = {"Player (P)": 0, "Banker (B)": 1, "Tie (T)": 2}

col1, col2, col3 = st.columns(3)
with col1:
    prev_1 = st.selectbox("ผลตาก่อนหน้า 1", list(option_map.keys()))
with col2:
    prev_2 = st.selectbox("ผลตาก่อนหน้า 2", list(option_map.keys()))
with col3:
    prev_3 = st.selectbox("ผลตาก่อนหน้า 3", list(option_map.keys()))

if st.button("ทำนายผลลัพธ์ถัดไป"):
    input_data = [[option_map[prev_1], option_map[prev_2], option_map[prev_3]]]
    prediction = model.predict(input_data)[0]
    result_decode = {0: "Player (P)", 1: "Banker (B)", 2: "Tie (T)"}
    st.success(f"ระบบคาดการณ์ว่า: **{result_decode[prediction]}**")
