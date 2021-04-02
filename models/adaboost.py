import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
from utils.harr import load_harr_features, get_pickled_key_features
from sys import stdout
from sklearn.metrics import confusion_matrix


class DecisionNode:
    def __init__(self):
        self.greater_boundary = None
        self.feature_index = None
        self.threshold = None
        self.weight = None

    def classify(self, data):
        feature_samples = data[:, self.feature_index]
        if self.greater_boundary:
            return feature_samples > self.threshold
        else:
            return feature_samples < self.threshold

    def signed_classification(self, data):
        feature_samples = data[:, self.feature_index]
        if self.greater_boundary:
            decisions = feature_samples > self.threshold
        else:
            decisions = feature_samples < self.threshold
        return self.weight * np.array([1 if decision else -1 for decision in decisions])

    def set_features(self, greater, threshold, idx):
        self.greater_boundary = greater
        self.threshold = threshold
        self.feature_index = idx

    def set_weight(self, weight):
        self.weight = weight


class Adaboost:
    def __init__(self, n_classifiers, feature_idx=None):
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.feature_idx = feature_idx
        self.train_acc = []
        self.val_acc = []

    def classify(self, data):
        decisions = np.squeeze(np.zeros((len(data), 1)))
        for classifier in self.classifiers:
            decisions += classifier.signed_classification(data)
        return decisions > 0

    def get_H_values(self, data):
        decisions = np.squeeze(np.zeros((len(data), 1)))
        for classifier in self.classifiers:
            decisions += classifier.signed_classification(data)
        return decisions

    def fit(self, data, labels, val_data=None, val_labels=None):
        if self.feature_idx is None:
            self.feature_idx = len(data[0])

        data_weights = np.ones(len(data)) / len(data)
        print("Fiting Adaboost Model")
        for i in range(self.n_classifiers):
            min_err = float('inf')
            node = DecisionNode()
            for feat_idx in self.feature_idx:
                feature_samples = data[:,feat_idx]
                for threshold in np.unique(feature_samples):
                    pred = feature_samples > threshold
                    total_err = np.sum(data_weights[pred != labels])

                    if total_err < min_err or 1-total_err < min_err:
                        node.set_features(total_err < (1-total_err), threshold, feat_idx)
                        min_err = np.minimum(total_err, 1-total_err)

            epsilon = 0.0001
            classifier_weight = .5 * np.log((1-min_err+epsilon)/(min_err+epsilon))
            node.set_weight(classifier_weight)
            pred = node.classify(data)
            results = np.array([1 if correct else -1 for correct in (pred == labels)])

            data_weights = data_weights * np.exp(-classifier_weight * results)
            data_weights = data_weights / sum(data_weights)
            self.classifiers.append(node)

            train_pred = self.classify(data)
            stdout.write('\rFinished '+ str(i+1)+ '/'+str(self.n_classifiers)+' classifiers. Training Accuracy: '+ str(np.mean(train_pred == labels)))
            self.train_acc.append(np.mean(train_pred == labels))

            if val_data is not None and val_labels is not None:
                val_pred = self.classify(val_data)
                acc = np.mean(val_pred == val_labels)
                stdout.write(" Test Accuracy: " + str(acc))
                self.val_acc.append(acc)

            stdout.write("  Classifier Weight: "+str(classifier_weight))

    def plot_training_values(self):
        epochs = np.arange(0,self.n_classifiers, 1)
        plt.plot(epochs, np.array(self.train_acc), label='Training Acc')
        plt.plot(epochs, np.array(self.val_acc), label="Validation Acc")
        plt.legend()
        plt.xlabel("Num classifiers")
        plt.ylabel("Accuracy")
        plt.title("Adaboost Fit Results")
        plt.show()


def train_ada_boost(n_feat=20):
    train_faces, train_background, test_faces, test_background = load_harr_features(split=.9)
    key_features = get_pickled_key_features(just_idx=True)[:n_feat]
    ada = Adaboost(n_feat, key_features)
    data = np.concatenate((train_faces, train_background))
    labels = np.squeeze(np.concatenate((np.ones((len(train_faces), 1)), np.zeros((len(train_background),1)))))
    val_data = np.concatenate((test_faces, test_background))
    val_labels = np.squeeze(np.concatenate((np.ones((len(test_faces), 1)), np.zeros((len(test_background), 1)))))
    ada.fit(data, labels, val_data, val_labels)
    ada.plot_training_values()
    tp = []
    fp = []
    H = ada.get_H_values(val_data)
    for threshold in np.unique(H):
        decisions = H >= threshold
        c_matrix = confusion_matrix(val_labels, decisions)
        fp.append(c_matrix[0,1]/sum(c_matrix[0]))
        tp.append(c_matrix[1,1]/sum(c_matrix[1]))

    plt.plot(np.array(fp),np.array(tp))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.title("Adaboost ROC")
    plt.show()


if __name__ == '__main__':
    train_ada_boost()