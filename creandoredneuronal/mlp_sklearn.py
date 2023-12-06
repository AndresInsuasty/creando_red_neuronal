import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Parámetros de generación del conjunto de datos
num_samples = 1000  # Número de muestras en el conjunto de datos
num_features = 2     # Número total de características
num_classes = 2      # Número de clases (binario en este ejemplo)
random_state = 42    # Semilla para reproducibilidad

# Asegúrate de que la suma de informativas, redundantes y repetidas sea menor que el número total de características
n_informative = 2    # Número de características informativas
n_redundant = 0      # Número de características redundantes
n_repeated = 0       # Número de características repetidas

# Crear el conjunto de datos
X, y = make_classification(n_samples=num_samples, n_features=num_features, n_informative=n_informative,
                           n_redundant=n_redundant, n_repeated=n_repeated, n_classes=num_classes,
                           n_clusters_per_class=1, random_state=random_state)

# Visualizar la distribución de las clases en el conjunto de datos
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.title('Distribución de clases en el conjunto de datos')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.colorbar(label='Clase')
plt.grid(True)
plt.savefig('img/distribucion_clases.png')
plt.show()

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Crear el clasificador de la red neuronal
clf = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=random_state)

# Entrenar el clasificador
history = clf.fit(X_train, y_train)

# Graficar la pérdida durante el entrenamiento
plt.figure(figsize=(8, 5))
plt.plot(history.loss_curve_)
plt.title('Curva de pérdida durante el entrenamiento')
plt.xlabel('Iteraciones de entrenamiento')
plt.ylabel('Pérdida')
plt.grid(True)
plt.savefig('img/curva_perdida.png')
plt.show()


# Predecir con el conjunto de prueba
predictions = clf.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, predictions)
print("Precisión del modelo:", accuracy)

reporte = classification_report(y_test, predictions)
print(reporte)

matriz_confusion = confusion_matrix(y_test, predictions)
print(matriz_confusion)