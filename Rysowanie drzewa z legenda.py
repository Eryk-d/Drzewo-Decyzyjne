import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# Plotting the decision tree
plt.figure(figsize=(12, 6))
plot_tree(decision_tree, feature_names=["Feature 1", "Feature 2", "Feature 3", "Feature 4"], filled=True)

# Create legend
class_names = ["Setosa", "Versicolor", "Virginica"]
class_colors = ["orange", "lightgreen", "#6d00ff"]
patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in class_colors]
plt.legend(patches, class_names, loc="center left", bbox_to_anchor=(1, 0.5))

plt.title("Decision Tree")
plt.tight_layout()
plt.show()
