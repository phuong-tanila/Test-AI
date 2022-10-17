import sklearn  # Thư viện train AI
from sklearn import linear_model  # Model AI có sẵn của sklearn
from sklearn.metrics import accuracy_score  # Hàm lấy độ chính xác trên tập Test
import pandas as pd  # Dùng chuyển file csv thành DataFrame
import numpy as np  # Dùng chuyển array thường thành vector
from sklearn.model_selection import train_test_split  # Dùng chia tập train và test

df = pd.read_csv('data.csv')

tmp = []
for i in df["diagnosis"].values:
    if i == "M":
        tmp.append(1)        # M thay bằng 1 (ác tính)

    else:
        tmp.append(0)        # B thay bằng 0 (lành tính)

# select những columns có kiểu float từ df
mylist = list(df.select_dtypes(include=['float']).columns)

# print(mylist)
del mylist[-1]  # Xóa cột cuối

y = np.asanyarray(tmp)         # Kết quả kỳ vọng (expected)
# Tìm ra kết quả thực tế (actual) để so sánh vs kỳ vọng
x = np.asanyarray(df[mylist])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, shuffle=False)
# Chia data thành 2 tập Train và Test (75% và 25%)


# Tạo 1 model LogisticRegression() từ thư viện sklearn
# Model Binary Classification (phân loại 2 class "M" và "B")
model = linear_model.LogisticRegression()
model.fit(x_train, y_train)               # Train tập trainset cho model

# Cho x_test vào model để nó ra thằng y_test_predict
y_test_predict = (model.predict(x_test))

# So sánh y_test_predict (actual) vs y_test (expected)
evaluate = accuracy_score(y_test, y_test_predict, normalize=False)
print("\nAccuracy:", str(evaluate) + "/" + str(len(y_test)),
      "(" + str(round(accuracy_score(y_test, y_test_predict)*100, 1))+"%)")
