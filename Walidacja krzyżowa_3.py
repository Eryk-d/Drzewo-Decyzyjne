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
min_samples_leaf_vec = np.array(range(1,25,1))
min_weight_fraction_leaf_vec = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

data = x
target = y_t
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_cr_md_split_leaf_weight_vec = np.zeros([len(min_samples_leaf_vec), len(min_weight_fraction_leaf_vec)])

for min_samples_leaf_ind in range(len(min_samples_leaf_vec)):
    for min_weight_fraction_leaf_ind in range(len(min_weight_fraction_leaf_vec)):
        PK_vec = np.zeros(CVN)
        for i, (train, test) in enumerate(skfold.split(data, target), start=0):
            x_train, x_test = data[train], data[test]
            y_train, y_test = target[train], target[test]

            decision_tree = DecisionTreeClassifier(random_state=0, max_depth=max_depth,
                                                   criterion=criterion,
                                                   splitter=splitter,
                                                   min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf_vec[min_samples_leaf_ind],
                                                   min_weight_fraction_leaf=min_weight_fraction_leaf_vec[
                                                       min_weight_fraction_leaf_ind])
            decision_tree = decision_tree.fit(x_train, y_train)
            result = decision_tree.predict(x_test)

            n_test_samples = test.size
            PK_vec[i] = np.sum(result == y_test) / n_test_samples

        PK_cr_md_split_leaf_weight_vec[min_samples_leaf_ind, min_weight_fraction_leaf_ind] = np.mean(PK_vec)
        print("min_samples_leaf: {} | min_weight_fraction_leaf: {} | PK: {}".format(
            min_samples_leaf_vec[min_samples_leaf_ind],
            min_weight_fraction_leaf_vec[min_weight_fraction_leaf_ind],
            PK_cr_md_split_leaf_weight_vec[min_samples_leaf_ind, min_weight_fraction_leaf_ind]))


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(min_samples_leaf_vec, min_weight_fraction_leaf_vec)
surf = ax.plot_surface(X, Y, PK_cr_md_split_leaf_weight_vec.T, cmap='viridis')
ax.set_xlabel('min_samples_leaf')
ax.set_ylabel('min_weight_fraction_leaf')
ax.set_zlabel('PK')
plt.show()
