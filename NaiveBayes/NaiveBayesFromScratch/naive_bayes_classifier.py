import numpy as np
import pandas as pd


def metrics_score(y_true, y_pred):

    # True Positive (TP): we predict a label of yes (positive), and the true label is yes.
    true_positive = np.sum(np.logical_and(y_pred == 'yes', y_true == 'yes'))

    # True Negative (TN): we predict a label of no (negative), and the true label is no.
    true_negative = np.sum(np.logical_and(y_pred == 'no', y_true == 'no'))

    # False Positive (FP): we predict a label of yes (positive), but the true label is no.
    false_positive = np.sum(np.logical_and(y_pred == 'yes', y_true == 'no'))

    # False Negative (FN): we predict a label of no (negative), but the true label is yes.
    false_negative = np.sum(np.logical_and(y_pred == 'no', y_true == 'yes'))

    # Accuracy: TP + TN / all samples
    accuracy = float(sum(y_pred == y_true)) / float(len(y_true))

    # False Positive Rate: Number of false positive over the sum of false positives and true negatives
    # The amount of falsely identified data as true from all the actual negative
    false_positive_rate = false_positive / (false_positive + true_negative)

    # False Negative Rate: Number of false negative over the sum of false negative and true positive
    # The amount of falsely identified data as negative from all the actual positive
    false_negative_rate = false_negative / (false_negative + true_positive)

    # Precision: How many retrieved items are relevant?
    precision = true_positive / (true_positive + false_positive)

    # Recall: How many relevant items are retrieved?
    recall = true_positive / (true_positive + false_negative)

    print(f"Accuracy is : {round(accuracy * 100, 2)}%")
    print(f"False Positive Rate is: {round(false_positive_rate * 100, 2)}%")
    print(f"False Negative Rate is: {round(false_negative_rate * 100, 2)}%")
    print(f"Precision is: {round(precision * 100, 2)}%")
    print(f"Recall is: {round(recall * 100, 2)}%")


def pre_processing(df):
    """ partioning data into features and target """

    X = df.drop([df.columns[-1]], axis=1)
    y = df[df.columns[-1]]

    return X, y


class NaiveBayes:
    """
        Bayes Theorem:
                                        Likelihood * Class prior probability
                Posterior Probability = -------------------------------------
                                            Predictor prior probability

                                           P(x|c) * p(c)
                               P(c|x) = ------------------
                                              P(x)
    """

    def __init__(self):

        """
            Attributes:
                likelihoods: Likelihood of each feature per class
                class_priors: Prior probabilities of classes
                pred_priors: Prior probabilities of features
                features: All features of dataset
        """
        self.features = list
        self.likelihoods = {}
        self.class_priors = {}

        self.X_train = np.array
        self.y_train = np.array
        self.train_size = int
        self.num_feats = int

    def fit(self, X, y):

        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]

        for feature in self.features:
            self.likelihoods[feature] = {}

            for feat_val in np.unique(self.X_train[feature]):
                for outcome in np.unique(self.y_train):
                    self.likelihoods[feature].update({str(feat_val) + '_' + str(outcome): 0})
                    self.class_priors.update({str(outcome): 0})

        self._calc_class_prior()
        self._calc_likelihoods()

    def _calc_class_prior(self):

        """ P(c) - Prior Class Probability """

        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    def _calc_likelihoods(self):

        """ P(x|c) - Likelihood """

        for feature in self.features:

            for outcome in np.unique(self.y_train):
                outcome_count = sum(self.y_train == outcome)
                feat_likelihood = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].value_counts().to_dict()

                for feat_val, count in feat_likelihood.items():
                    self.likelihoods[feature][feat_val + '_' + outcome] = count / outcome_count

    def predict(self, X):

        """ Calculates Posterior probability P(c|x) """

        results = []
        X = np.array(X)

        for query in X:
            probs_outcome = {}
            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1

                for feat, feat_val in zip(self.features, query):
                    likelihood *= self.likelihoods[feat][feat_val + '_' + outcome]

                posterior = (likelihood * prior)

                probs_outcome[outcome] = posterior

            result = max(probs_outcome, key=lambda x: probs_outcome[x])
            results.append(result)

        return np.array(results)


if __name__ == "__main__":
    # Weather Dataset
    print("\nWeather Dataset:")

    df = pd.read_csv("data/weather.csv")
    # print(df)

    # Split fearures and target
    X, y = pre_processing(df)

    nb_clf = NaiveBayes()
    nb_clf.fit(X, y)

    metrics_score(y, nb_clf.predict(X))

    # Query 1:
    query = np.array([['Rainy', 'Mild', 'Normal', 't']])
    print("Query 1:- {} ---> {}".format(query, nb_clf.predict(query)))

    # Query 2:
    query = np.array([['Overcast', 'Cool', 'Normal', 't']])
    print("Query 2:- {} ---> {}".format(query, nb_clf.predict(query)))

    # Query 3:
    query = np.array([['Sunny', 'Hot', 'High', 't']])
    print("Query 3:- {} ---> {}".format(query, nb_clf.predict(query)))