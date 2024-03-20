import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Membaca data dari file CSV

data = pd.read_csv('DATA DBD.csv')

# Memisahkan fitur dan label
X = data.drop(columns=['Gejalah'])  # Fitura
y = data['Gejalah']  # Label

# Memasukkan fitur gejala ke dalam fitur-fitur
gejala = data['Gejalah']  # Ambil kolom gejala dari dataset
X['Gejala'] = gejala  # Tambahkan kolom gejala ke dalam fitur

# Melakukan one-hot encoding untuk fitur kategorikal jika diperlukan
X = pd.get_dummies(X)

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Membuat model Decision Tree
model = DecisionTreeClassifier()

# Melatih model
model.fit(X_train, y_train)

# Memprediksi data uji
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Mendapatkan class names dari model
class_names = model.classes_

# Visualisasi decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=class_names)
plt.savefig('decision_tree.png')  # Menyimpan visualisasi ke file
plt.show()