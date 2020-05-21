
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


class Analize:
    def __init__(self, X_train, y_train):
        self.model = LogisticRegression(solver='lbfgs')
        self.model = self.model.fit(X_train, y_train.values.ravel())



    def evaluate_model(self, X_test, y_test):
        accuracy = self.model.score(X_test, y_test)
        predictions = self.model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, predictions)
        print(classification_report(y_test, predictions))
        print(conf_matrix)
        print(accuracy)

    def testing_model(self, input,):
        result = self.model.predict(input)
        return result

    def testing_model_prob(self, input,):
        result = self.model.predict_proba(input)
        return result