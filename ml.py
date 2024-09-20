from google.colab import drive
drive.mount('/content/drive')

!pip install -U scikit-learn

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ai_data.csv")
df.describe()

x = df.drop(columns=["Active-inactive prediction"])
y = df["Active-inactive prediction"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

score = accuracy_score(y_test, predictions)
print("Mean Accuracy:", score)

# Making predictions on new data
b = [269, 79, 107, 417, 67]
p = model.predict([b])
print("Predictions of the query:", p[0])

b = [400, 200, 30, 280, 130]
p = model.predict([b])
print("Predictions of the query:", p[0])

b = [370, 220, 49, 300, 100]
p = model.predict([b])
print("Predictions of the query:", p[0])

b = [400, 200, 30, 300, 120]
p = model.predict([b])
print("Predictions of the query:", p[0])

b = [200, 50, 110, 400, 70]
p = model.predict([b])
print("Predictions of the query:", p[0])
