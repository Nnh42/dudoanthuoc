import tkinter as tk
from tkinter import ttk
from Perceptron import Perceptron
import numpy as np
from PIL import Image, ImageTk

# ✅ Load model đã huấn luyện
pct = Perceptron()
pct.load_model("perceptron_model.pkl")

# Hàm dự đoán
def predict_drug():
    try:
        sex = 1 if sex_var.get() == "M" else 0
        bp = {"LOW": 3, "NORMAL": 4, "HIGH": 2}[bp_var.get()]
        cholesterol = {"LOW": 3, "NORMAL": 4, "HIGH": 2}[cholesterol_var.get()]
        na_to_k = float(na_to_k_entry.get() or 0)
        age = float(age_entry.get() or 0)

        input_data = np.array([[age, sex, bp, cholesterol, na_to_k]])
        prediction = pct.predict(input_data)
        predicted_drug = str(prediction[0])  # ✅ không cần ép int

        result_label.config(text=f"Kết quả dự đoán: {predicted_drug}", fg="green")

        # Hiển thị hình ảnh thuốc
        try:
            if predicted_drug == "drugX":
                img = Image.open("drugX.jpg")
            elif predicted_drug == "drugY":
                img = Image.open("drugY.jpg")
            elif predicted_drug == "drugZ":
                img = Image.open("drugZ.jpg")
            else:
                drug_image_label.config(image='', text="Không có hình ảnh")
                return

            img = img.resize((150, 100), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            drug_image_label.config(image=photo, text=predicted_drug)
            drug_image_label.image = photo
        except FileNotFoundError:
            drug_image_label.config(text=f"Hình ảnh cho {predicted_drug} chưa có", image='')

    except Exception as e:
        result_label.config(text=f"Lỗi: {str(e)}", fg="red")

# Giao diện
window = tk.Tk()
window.title("Dự đoán loại thuốc bằng Perceptron")
window.geometry("400x500")

tk.Label(window, text="Dự đoán loại thuốc", font=("Arial", 14, "bold")).pack(pady=10)

tk.Label(window, text="Giới tính (Sex):").pack()
sex_var = tk.StringVar(value="M")
ttk.Combobox(window, textvariable=sex_var, values=["M", "F"], state="readonly").pack()

tk.Label(window, text="Huyết áp (BP):").pack()
bp_var = tk.StringVar(value="LOW")
ttk.Combobox(window, textvariable=bp_var, values=["LOW", "NORMAL", "HIGH"], state="readonly").pack()

tk.Label(window, text="Cholesterol:").pack()
cholesterol_var = tk.StringVar(value="LOW")
ttk.Combobox(window, textvariable=cholesterol_var, values=["LOW", "NORMAL", "HIGH"], state="readonly").pack()

tk.Label(window, text="Na_to_K:").pack()
na_to_k_entry = tk.Entry(window)
na_to_k_entry.pack()
na_to_k_entry.insert(0, "25.0")

tk.Label(window, text="Tuổi:").pack()
age_entry = tk.Entry(window)
age_entry.pack()
age_entry.insert(0, "55")

tk.Button(window, text="Dự đoán", command=predict_drug, bg="lightblue").pack(pady=10)
result_label = tk.Label(window, text="", font=("Arial", 12))
result_label.pack(pady=10)

drug_image_label = tk.Label(window, text="Hình ảnh thuốc", compound=tk.TOP)
drug_image_label.pack(pady=10)

window.mainloop()
