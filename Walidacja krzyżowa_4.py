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
max_features_vec = np.array(["sqrt", "log2"])
max_leaf_nodes_vec = np.array(range(2, 15, 1))

data = x
target = y_t
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_cr_md_split_leaf_weight_feat_node_vec = np.zeros([len(max_features_vec), len(max_leaf_nodes_vec)])

for feat_ind, max_features in enumerate(max_features_vec):
    for node_ind, max_leaf_nodes in enumerate(max_leaf_nodes_vec):
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
                                                   max_leaf_nodes=max_leaf_nodes)
            decision_tree = decision_tree.fit(x_train, y_train)
            result = decision_tree.predict(x_test)

            n_test_samples = test.size
            PK_vec[i] = np.sum(result == y_test) / n_test_samples

        PK_cr_md_split_leaf_weight_feat_node_vec[feat_ind, node_ind] = np.mean(PK_vec)
        print("max_features: {} | max_leaf_nodes: {} | PK: {}".format(
            max_features,
            max_leaf_nodes,
            PK_cr_md_split_leaf_weight_feat_node_vec[feat_ind, node_ind]))

# Etykiety osi X i Y
X_labels = max_features_vec
Y_labels = max_leaf_nodes_vec.astype(str)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(len(X_labels)), range(len(Y_labels)))
surf = ax.plot_surface(X, Y, PK_cr_md_split_leaf_weight_feat_node_vec.T, cmap='viridis')
ax.set_xlabel('max_features')
ax.set_ylabel('max_leaf_nodes')
ax.set_xticks(range(len(X_labels)))
ax.set_yticks(range(len(Y_labels)))
ax.set_xticklabels(X_labels)
ax.set_yticklabels(Y_labels)
ax.set_zlabel('PK')
plt.show()
