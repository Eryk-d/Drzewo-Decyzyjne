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
splitter_vec = ["best", "random"]
min_samples_split_vec = np.array([2, 5, 10, 20])

data = x
target = y_t
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_cr_md_split_vec = np.zeros([len(splitter_vec), len(min_samples_split_vec)])

for splitter_ind in range(len(splitter_vec)):
    for min_samples_split_ind in range(len(min_samples_split_vec)):
        PK_vec = np.zeros(CVN)
        for i, (train, test) in enumerate(skfold.split(data, target), start=0):
            x_train, x_test = data[train], data[test]
            y_train, y_test = target[train], target[test]

            decision_tree = DecisionTreeClassifier(random_state=0, max_depth=max_depth, criterion=criterion,
                                                   splitter=splitter_vec[splitter_ind],
                                                   min_samples_split=min_samples_split_vec[min_samples_split_ind])
            decision_tree = decision_tree.fit(x_train, y_train)
            result = decision_tree.predict(x_test)

            n_test_samples = test.size
            PK_vec[i] = np.sum(result == y_test) / n_test_samples

        PK_cr_md_split_vec[splitter_ind, min_samples_split_ind] = np.mean(PK_vec)
        print("splitter: {} | min_samples_split: {} | PK: {}".format(splitter_vec[splitter_ind],
                                                                      min_samples_split_vec[min_samples_split_ind],
                                                                      PK_cr_md_split_vec[
                                                                          splitter_ind, min_samples_split_ind]))


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(len(splitter_vec)), min_samples_split_vec)
surf = ax.plot_surface(X, Y, PK_cr_md_split_vec.T, cmap='viridis')
ax.set_xlabel('splitter')
ax.set_xticks(np.array(range(len(splitter_vec))))
ax.set_xticklabels(splitter_vec)
ax.set_ylabel('min_samples_split')
ax.set_zlabel('PK')
plt.colorbar(surf, ax=ax, label='PK')
plt.show()


