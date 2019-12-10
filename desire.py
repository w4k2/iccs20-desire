"""
Dynamic Ensemble Selection using Imbalance Ratio and Euclidean Distance
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import sys
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(threshold=sys.maxsize)

class DESIRE(BaseEstimator, ClassifierMixin):
    """
    TBA
    """

    def __init__(self, ensemble=[], k=7, random_state=42, w=0.1, mode="whole"):
        self.ensemble = ensemble
        self.random_state = random_state
        self.k = k
        self.w = w
        self.mode=mode

    def fit(self, X, y):
        # information about data distribution
        maj, min = np.bincount(y.astype(int))
        if maj < min:
            self.min = maj/min
        else:
            self.min = min/maj
        self.maj = 1-self.min
        self.ir = self.maj/self.min
        print(self.min)
        print(self.maj)

        self.X_dsel = X
        self.y_dsel = y
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(self.X_dsel, self.y_dsel)

    def estimate_competence(self, X):
        self.competences = np.zeros((len(self.ensemble), X.shape[0], 2))
        self.distance, self.neighbors = self.knn.kneighbors(X=X, n_neighbors=self.k)

        local_X = self.X_dsel[self.neighbors,:]
        local_y = self.y_dsel[self.neighbors]

        for i, instance in enumerate(local_X):
            for j, base_classifier in enumerate(self.ensemble):
                pred = base_classifier.predict(instance)

                for k in range(len(pred)):
                    if self.mode == "whole":
                        if pred[k] == local_y[i][k] == 0:
                            self.competences[j,i,0] = (self.competences[j,i,0] * self.w) + self.distance[i][k]# * self.w
                        elif pred[k] == local_y[i][k] == 1:
                            self.competences[j,i,1] = (self.competences[j,i,1] * self.w) + self.distance[i][k] * self.ir# * self.w
                        elif pred[k] == 0 and local_y[i][k] == 1:
                            self.competences[j,i,1] = (self.competences[j,i,1] * self.w) - self.distance[i][k]# * self.w
                        elif pred[k] == 1 and local_y[i][k] == 0:
                            self.competences[j,i,0] = (self.competences[j,i,0] * self.w) - self.distance[i][k] * self.ir#W * self.w
                    # if self.mode == "whole":
                    #     if pred[k] == local_y[i][k] == 0:
                    #         self.competences[j,i,0] = (self.competences[j,i,0]) + (self.distance[i][k] * self.w)
                    #     elif pred[k] == local_y[i][k] == 1:
                    #         self.competences[j,i,1] = (self.competences[j,i,1]) + (self.distance[i][k] * self.ir * self.w)
                    #     elif pred[k] == 0 and local_y[i][k] == 1:
                    #         self.competences[j,i,1] = (self.competences[j,i,1]) - (self.distance[i][k] * self.w)
                    #     elif pred[k] == 1 and local_y[i][k] == 0:
                    #         self.competences[j,i,0] = (self.competences[j,i,0]) - (self.distance[i][k] * self.ir * self.w)
                    elif self.mode == "correct":
                        if pred[k] == local_y[i][k] == 0:
                            self.competences[j,i,0] = (self.competences[j,i,0]) + self.distance[i][k]# * self.min)
                        elif pred[k] == local_y[i][k] == 1:
                            self.competences[j,i,1] = (self.competences[j,i,1]) + self.distance[i][k]# * self.maj)
                    elif self.mode == "wrong":
                        if pred[k] == local_y[i][k] == 0:
                           self.competences[j,i,0] = (self.competences[j,i,0]) + self.distance[i][k]# * self.min)
                        elif pred[k] == local_y[i][k] == 1:
                           self.competences[j,i,1] = (self.competences[j,i,1]) + self.distance[i][k]# * self.maj)
                        elif pred[k] == 0 and local_y[i][k] == 1:
                            self.competences[j,i,1] = (self.competences[j,i,1]) - self.distance[i][k]# * self.min)
                        elif pred[k] == 1 and local_y[i][k] == 0:
                            self.competences[j,i,0] = (self.competences[j,i,0]) - self.distance[i][k]# * self.maj)

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble])

    def predict(self, X):
        self.estimate_competence(X)
        esm = self.ensemble_support_matrix(X)
        for i in range(self.competences.shape[0]):
            scaler = MinMaxScaler()
            self.competences[i] = scaler.fit_transform(self.competences[i])
        esm *= self.competences

        average_support = np.max(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        return prediction

    def score(self, X, y):
        return balanced_accuracy_score(y, self.predict(X))
