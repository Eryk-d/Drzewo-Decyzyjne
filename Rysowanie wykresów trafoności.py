import numpy as np
from sklearn.tree import DecisionTreeClassifier
import hickle as hkl
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
min_impurity_decrease = 0
ccp_alpha = 0

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
decision_tree.fit(x, y_t)
result = decision_tree.predict(x)

classification_vector = (result == y_t).astype(int)

plt.figure()
plt.plot(range(len(classification_vector)), classification_vector, 'bo', markersize=4)
plt.xlabel('Record')
plt.ylabel('Correct Classification')
plt.title('Correct Classification for Each Record')
plt.ylim([-0.1, 1.1])
plt.grid(True)
plt.show()
