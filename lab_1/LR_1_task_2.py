import numpy as np
from sklearn import preprocessing

# Надання вхідних даних
input_data = np.array([[4.3, -9.9, -3.5],
                       [-2.9, 4.1, 3.3],
                       [-2.2, 8.8, -6.1],
                       [3.9, 1.4, 2.2]])

# Бінаризація даних
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\n Binarized data:\n", data_binarized)

# Виведення середнього значення та стандартного відхилення
print("\nBEFORE: ")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Виключення середнього
data_scaled = preprocessing.scale(input_data)
print("\nAFTER: ")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# Масштабування MinMax
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nМin max scaled data:\n", data_scaled_minmax)

# Нормалізація даних
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nl1 normalized data:\n", data_normalized_l1)
print("\nl2 normalized data:\n", data_normalized_l2)

# Створення кодувальника та встановлення відповідності
# між мітками та числами
encoder = preprocessing.LabelEncoder()
encoder.fit(input_data.ravel())

# Виведення відображення
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# Перетворення міток за допомогою кодувальника
test_labels = [[4.3, -3.5, 4.3],
               [-9.9, 8.8, 1.4]]
test_labels_array = np.array(test_labels)
encoded_values = encoder.transform(test_labels_array.flat)
print("\nLabels =", test_labels)
print("Encoded values =", encoded_values.reshape(test_labels_array.shape))

# Декодування набору чисел за допомогою декодера
encoded_values = [6, 1, 0, 2, 5, 4, 3, 7, 8]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", decoded_list)
