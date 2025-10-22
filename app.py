import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import warnings

# Cấu hình giao diện
st.set_page_config(page_title="Drug Prediction Demo", page_icon="💊", layout="centered")
st.title("💊 Dự đoán loại thuốc")
st.write("Nhập thông tin bệnh nhân bên dưới để dự đoán loại thuốc phù hợp.")

# Hàm mã hóa dữ liệu đầu vào
def data_encoder(X):
    X = X.copy()
    for col in X.columns:
        X[col] = X[col].replace({
            "F": 0, "M": 1,
            "HIGH": 2, "LOW": 3, "NORMAL": 4
        })
    return X

# Load mô hình SVM
model_path = "svm_model.pkl"
if not os.path.exists(model_path):
    st.error(f"❌ Không tìm thấy file mô hình tại {model_path}. Vui lòng chạy lại file drug_solution.py để lưu mô hình.")
    st.stop()
model = joblib.load(model_path)
st.success(f"✅ Đã load mô hình SVM từ {model_path}")

# Load LabelEncoder
le_path = "label_encoder.pkl"
if not os.path.exists(le_path):
    st.error(f"❌ Không tìm thấy file LabelEncoder tại {le_path}. Vui lòng chạy lại file drug_solution.py để lưu LabelEncoder.")
    st.stop()
le_y = joblib.load(le_path)
st.success(f"✅ Đã load LabelEncoder từ {le_path}")

# Form nhập liệu
st.subheader("🧍‍♂️ Nhập thông tin bệnh nhân:")
col1, col2 = st.columns(2)
sex = col1.selectbox("Giới tính (Sex)", ["M", "F"])
bp = col2.selectbox("Huyết áp (BP)", ["LOW", "NORMAL", "HIGH"])
chol = col1.selectbox("Cholesterol", ["NORMAL", "HIGH"])
age = col2.number_input("Tuổi", min_value=0, max_value=120, value=35)
na = col1.number_input("Na_to_K (Tỉ lệ Na/K)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)

# Nút dự đoán
if st.button("🔮 Dự đoán"):
    if age < 17:
        st.success("💊 Kết quả dự đoán: **DrugZ** (Thuốc đặc biệt cho người dưới 17 tuổi)")
        try:
            img_path = pathlib.Path(__file__).parent / "images" / "drugz.jpg"
            if img_path.exists():
                st.image(str(img_path), caption="Minh họa: Thuốc DrugZ", width=250)
            else:
                st.info("Không tìm thấy hình minh họa DrugZ.")
            #st.image(image, caption="Minh họa: Thuốc DrugZ", width=250)
        except:
            st.info("Không tìm thấy hình minh họa DrugZ.")
    else:
        # Tạo DataFrame đầu vào
        input_df = pd.DataFrame([[age, sex, bp, chol, na]], 
                                columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
        # Mã hóa dữ liệu
        input_encoded = data_encoder(input_df).astype(float)
        # Dự đoán
        pred = model.predict(input_encoded)[0]
        # Giải mã nhãn
        try:
            drug_name = le_y.inverse_transform([pred])[0]
        except ValueError as e:
            st.error(f"❌ Lỗi khi giải mã nhãn: {e}")
            st.stop()

        st.success(f"💊 Kết quả dự đoán: **{drug_name}**")

        try:
            import pathlib
            img_path = pathlib.Path(__file__).parent / "images" / f"{drug_name.lower()}.jpg"
            if img_path.exists():
                st.image(str(img_path), caption=f"Minh họa: Thuốc {drug_name}", width=250)
            else:
                st.info("Không có hình minh họa cho loại thuốc này.")

            #st.image(image, caption=f"Minh họa: Thuốc {drug_name}", width=250)
        except:
            st.info("Không có hình minh họa cho loại thuốc này.")

st.markdown("---")
st.caption("🧠 Mô hình Support Vector Machine ")
