# model_dispatcher.py

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

models = {
    "decision_tree_gini": DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": DecisionTreeClassifier(criterion="entropy"),
    "log_reg": LogisticRegression(),
    "k_neighbors": KNeighborsClassifier(),
    "svc": SVC(),
    "random_forest": RandomForestClassifier(),
    "adaboost": AdaBoostClassifier(),
    "naive_bayes": GaussianNB(),
}
