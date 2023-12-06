"""
Este script genera un dataset de clasificacion binaria
"""

from sklearn.datasets import make_classification


# Parametros

N_SAMPLES = 10
N_FEATURES = 2
N_CLASSES = 2
RANDOM_STATE = 123

N_INFORMATIVE = 2
N_REDUNDANT = 0
N_REPEATED = 0

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=N_REDUNDANT,
    n_repeated=N_REPEATED,
    n_classes=N_CLASSES,
    n_clusters_per_class=1,
    random_state=RANDOM_STATE)

print(X)
print("*"*50)
print(y)
