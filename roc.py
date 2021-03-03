import numpy as np
from scipy.special import logsumexp
from pathing import *
import matplotlib.pyplot as plt
from os import path
import pickle


def test_models(face_model, background_model, face_set, back_set, title):
    print("Testing Models")
    p_face = []
    p_back = []
    test_set = np.concatenate((face_set, back_set))
    labels = np.concatenate((np.ones(len(face_set)), np.zeros((len(back_set))))).astype('bool')
    for image in test_set:
        p_face.append(face_model.logpdf(image))
        p_back.append(background_model.logpdf(image))
    plot_roc(np.array(p_face), np.array(p_back), labels, title)


def plot_roc(p_face, p_background, labels, title, num_points=100):
    tp = []
    fp = []
    max = np.max((np.max(p_face), np.max(p_background))) + 5
    min = np.min((np.min(p_face), np.min(p_background))) - 5
    thresholds = np.geomspace(min, max, num=num_points)
    for prob in thresholds:
        decisions = calculate_decisions(p_face,p_background, prob)
        tp_i, fp_i = calculate_rates(decisions, labels)
        tp.append(tp_i)
        fp.append(fp_i)

    plt.scatter(fp,tp)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title(title)
    plt.show()


def calculate_decisions(p_face, p_background, p_threshold):
    decisions = []
    prior = np.log(.5)
    for face, back in zip(p_face, p_background):
        p = face + prior - logsumexp((face+prior, back+prior))
        decisions.append(p > p_threshold)
    return np.array(decisions)


def calculate_rates(decisions, labels):
    tp = np.sum(decisions & labels) / np.sum(labels)
    fp = np.sum(decisions & np.logical_not(labels))/ np.sum(np.logical_not(labels))
    return tp, fp


def option_to_save_model():
    print("Would you like to save this model (y/n)?")
    ans = input()
    if ans == 'n':
        return
    else:
        print("What would you like to save model as?")
        name = input()


def imshow_sample(sample):
    sample = sample.reshape((20,20,3))/np.max(sample)
    plt.imshow(sample)
    plt.show()


def plot_changes(mc, cc, title):
    plt.figure()
    plt.plot(range(len(mc)), mc)
    plt.plot(range(len(cc)), cc)
    plt.ylabel("Avg Change")
    plt.xlabel("Step")
    plt.title(title)
    plt.legend(("Mean Change", "Cov Change"))
    plt.show()


def net_change(x,y ,threshold=1):
    change = np.sum(np.abs(x-y))/x.size
    return change > threshold, change
