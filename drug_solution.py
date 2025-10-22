import numpy as np
import pandas as pd
#import warnings
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import LogisticRegression
from Perceptron import Perceptron
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ✅ Đọc dữ liệu
df = pd.read_csv('Data/drugZ_random.csv')

# ✅ Mã hóa các cột kiểu chuỗi bằng LabelEncoder
for col in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df['Drug'].values


# ✅ Tách dữ liệu đầu vào và đầu ra
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df['Drug'].values

# ✅ Chia dữ liệu train/test (cố định random_state để kết quả ổn định)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=50
)

# ✅ Khởi tạo mô hình
pct = Perceptron(learning_rate=0.01, epochs=300)  # tăng epochs cho Perceptron hội tụ tốt hơn

svm = SVC(kernel='rbf', C=1, gamma='scale')       # dùng kernel 'rbf' để mô hình phi tuyến tốt hơn


#tree = DecisionTreeClassifier(
#criterion='entropy', max_depth=4, min_samples_leaf=2, min_samples_split=2, random_state=42)

#neural = MLPClassifier(
#   max_iter=1000, alpha=0.0001, learning_rate_init=0.01, solver='adam', activation='relu', random_state=42)

#logistic = LogisticRegression(
#penalty='l2', solver='lbfgs', max_iter=500, C=1, random_state=42)

# ✅ Huấn luyện
pct.fit(X_train, y_train)
perceptron_y_pred = pct.predict(X_test)

svm.fit(X_train, y_train)
svm_y_pred = svm.predict(X_test)

#tree.fit(X_train, y_train)
#tree_y_pred = tree.predict(X_test)

#neural.fit(X_train, y_train)
#neural_y_pred = neural.predict(X_test)

#logistic.fit(X_train, y_train)
#logistic_y_pred = logistic.predict(X_test)


# ✅ Tính toán các chỉ số đánh giá
def evaluate_model(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted'),
        recall_score(y_true, y_pred, average='weighted'),
        f1_score(y_true, y_pred, average='weighted'),
    )

results = {
    "Perceptron": evaluate_model(y_test, perceptron_y_pred),
    "SVM": evaluate_model(y_test, svm_y_pred),
    #"DecisionTree": evaluate_model(y_test, tree_y_pred),
    #"NeuralNetwork": evaluate_model(y_test, neural_y_pred),
    #"LogisticRegression": evaluate_model(y_test, logistic_y_pred),
}

# ✅ Hiển thị kết quả ra bảng
df_result = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1-score"])
print("\n=== 📊 Kết quả đánh giá mô hình ===\n")
print(df_result.round(3))

# ✅ Vẽ biểu đồ so sánh
labels = list(results.keys())
metrics = np.array(list(results.values())).T
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5 * width, metrics[0], width, label='Accuracy')
ax.bar(x - 0.5 * width, metrics[1], width, label='Precision')
ax.bar(x + 0.5 * width, metrics[2], width, label='Recall')
ax.bar(x + 1.5 * width, metrics[3], width, label='F1 Score')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_title("So sánh hiệu năng các mô hình (dữ liệu DrugZ)")

# Hiển thị giá trị trên đầu cột
for i, v in enumerate(metrics[0]):
    ax.text(i - 1.5 * width, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()
