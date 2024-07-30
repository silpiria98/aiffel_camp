# 데이터 로드
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True, as_frame=True)

# 분할
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

# 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_valid = scaler.transform(X_valid)

# 학습
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(scaled_X_train, y_train)

train_pred = classifier.predict(scaled_X_train)
valid_pred = classifier.predict(scaled_X_valid)

from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# 저장
import joblib
joblib.dump(scaler, "scaler.joblib")
joblib.dump(classifier, "classifier.joblib")