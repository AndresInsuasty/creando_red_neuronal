from sklearn.datasets import make_classification


# Parametros

n_samples = 10
n_features = 2
n_classes = 2
random_state = 123

n_informative = 2
n_redundant = 0
n_repeated = 0

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_repeated=n_repeated,
    n_classes=n_classes,
    n_clusters_per_class=1,
    random_state=random_state)

print(X)
print("*"*50)
print(y)