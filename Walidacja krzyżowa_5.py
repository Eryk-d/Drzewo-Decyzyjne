import numpy as np
from sklearn.tree import DecisionTreeClassifier
import hickle as hkl
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

x, y_t, x_norm, x_n_s, y_t_s = hkl.load('iris.hkl')
y_t -= 1
x = x.T
y_t = np.squeeze(y_t)

criterion = "entropy"
max_depth = 3
splitter = "random"
min_samples_split = 5
min_samples_leaf = 2
min_weight_fraction_leaf = 0
max_features = "log2"
max_leaf_nodes = 5
min_impurity_decrease_vec = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ccp_alpha_vec = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

data = x
target = y_t
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_imp_alpha_vec = np.zeros((len(min_impurity_decrease_vec), len(ccp_alpha_vec)))

for imp_ind, min_impurity_decrease in enumerate(min_impurity_decrease_vec):
    for alpha_ind, ccp_alpha in enumerate(ccp_alpha_vec):
        PK_vec = np.zeros(CVN)
        for i, (train, test) in enumerate(skfold.split(data, target), start=0):
            x_train, x_test = data[train], data[test]
            y_train, y_test = target[train], target[test]

            decision_tree = DecisionTreeClassifier(random_state=0, max_depth=max_depth,
                                                   criterion=criterion,
                                                   splitter=splitter,
                                                   min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf,
                                                   min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                   max_features=max_features,
                                                   max_leaf_nodes=max_leaf_nodes,
                                                   min_impurity_decrease=min_impurity_decrease,
                                                   ccp_alpha=ccp_alpha)
            decision_tree = decision_tree.fit(x_train, y_train)
            result = decision_tree.predict(x_test)

            n_test_samples = test.size
            PK_vec[i] = np.sum(result == y_test) / n_test_samples

        PK_imp_alpha_vec[imp_ind, alpha_ind] = np.mean(PK_vec)
        print("min_impurity_decrease: {} | ccp_alpha: {} | PK: {}".format(
            min_impurity_decrease,
            ccp_alpha,
            PK_imp_alpha_vec[imp_ind, alpha_ind]))

# Etykiety osi X i Y
X_labels = min_impurity_decrease_vec.astype(str)
Y_labels = ccp_alpha_vec.astype(str)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(len(X_labels)), range(len(Y_labels)))
surf = ax.plot_surface(X, Y, PK_imp_alpha_vec.T, cmap='viridis')
ax.set_xlabel('min_impurity_decrease')
ax.set_ylabel('ccp_alpha')
ax.set_xticks(range(len(X_labels)))
ax.set_yticks(range(len(Y_labels)))
ax.set_xticklabels(X_labels)
ax.set_yticklabels(Y_labels)
ax.set_zlabel('PK')
plt.show()
