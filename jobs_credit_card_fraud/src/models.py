from typing import Any, Dict, List
import pandas

# Let's implement simple classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Use GridSearchCV to find the best parameters
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

import nbox

class ModelSearching(nbox.Operator):
  classifiers = {
    "LogisiticRegression": LogisticRegression,
    "KNearest": KNeighborsClassifier,
    "Support Vector Classifier": SVC,
    "DecisionTreeClassifier": DecisionTreeClassifier
  }

  def __init__(self) -> None:
    super().__init__()

  def forward(self, X_train: pandas.DataFrame, y_train: pandas.DataFrame) -> str:
    # Our data is already scaled we should split our training and test sets

    # Train multiple models and evaluate their performance
    highest_score_classifier = None
    highest_accuracy = 0
    for classifier_name, classifier in self.classifiers.items():
      classifier.fit(X_train, y_train)
      training_score = cross_val_score(classifier, X_train, y_train, cv=5)

      acc = training_score.mean()
      if acc > highest_accuracy:
        highest_accuracy = acc
        highest_score_classifier = classifier_name
      print("Model:", classifier_name, "Accuracy:", acc)
  
    return highest_score_classifier, highest_accuracy


class GridSearchTrainer(nbox.Operator):
  def __init__(
    self,
    model_to_search_params: Dict[str, List[Any]]
  ) -> None:
    super().__init__()
    self.model_searcher = ModelSearching()
    self.model_to_search_params = model_to_search_params

  @staticmethod
  def _grid_search(model_cls, search_params, X_train, y_train):
    grid = GridSearchCV(model_cls(), search_params)
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_
    return best_estimator

  def forward(self, X: pandas.DataFrame, y: pandas.DataFrame, model_name: str = "all", find_model: bool = False):
    # This is explicitly used for undersampling.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Turn the values into an array for feeding the classification algorithms.
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    # find the best model
    if find_model:
      model_name, model_acc = self.model_searcher()
      print("Chosen Model:", model_name, "Accuracy:", model_acc)
      models = {
        model_name: ModelSearching.classifiers[model_name]
      }
    elif model_name != "all":
      models = {
        model_name: ModelSearching.classifiers[model_name]
      }
    else:
      models = ModelSearching.classifiers

    # iterate over the models and log scores for each one
    best_model = None
    best_model_name = None
    best_score = -1
    for model_name, model_cls in models.items():
      print("Training model:", model_name)
      model = self._grid_search(model_cls, self.model_to_search_params[model_name], X_train, y_train)
      score = cross_val_score(model, X_train, y_train, cv=5).mean()
      print(f'{model_name} Cross Validation Score: ', round(score * 100, 2).astype(str) + '%')

      if score > best_score:
        best_score = score
        best_model = model
        best_model_name = model_name

    return best_model, best_model_name
