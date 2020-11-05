# import modules
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Memanggila Dataset
bunga_iris = datasets.load_iris()

print(type(bunga_iris))             # Type
print(bunga_iris.data.shape)        # Jumlah baris dan kolom
print(bunga_iris.target_names)      # Target sets dari data

# training dataset
X = bunga_iris.data
# target set
Y = bunga_iris.target

## Konversi dari datasets ke DataFrame
iris_df = pd.DataFrame(X, columns=bunga_iris.feature_names)

# Print 5 data pertama
print(iris_df.head())

## Memanggil KNN dan Melakukan Training
knn_algo = KNeighborsClassifier( n_neighbors=6,
                                 weights='uniform',
                                 algorithm='auto',
                                 metric='euclidean' )
# Training data 
X_train = bunga_iris['data']
y_train = bunga_iris['target']
knn_algo.fit(X_train, y_train)

# data yang akan prediksi
### data_baru[[sepal_length, sepal_width, petal_length, petal_width]]
data_baru = [[1.2, 3.5, 1.2, 2.4]]
# melakukan prediksi
Y_pred = knn_algo.predict(data_baru)

if Y_pred == 0:
    print('Hasil Prediksi = Jenis Sentosa')
elif Y_pred == 1:
    print('Hasil Prediksi = Jenis Versicolor')
elif Y_pred == 2:
    print('Hasil Prediksi = Jenis Virginica')
else:
    print('Keyword Error!!')















