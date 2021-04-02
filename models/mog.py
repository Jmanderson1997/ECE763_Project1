import numpy as np
import os
from utils.pathing import *
from models.gaussian import Gaussian
from scipy.special import logsumexp
from dataset.data import get_pickled_data
from utils.roc import test_models, imshow_sample, net_change


class MixtureGaussian:
    def __init__(self, n_gaussian, samples):
        self.pdfs = []
        self.weights = np.repeat(1/n_gaussian, n_gaussian)
        for n in range(n_gaussian):
            sub_samples = samples[np.random.choice(range(0,len(samples)), size=1500)]
            self.pdfs.append(Gaussian(samples=sub_samples))

    def run_em(self, data):
        cont = True
        n = 0
        mc = []
        cc = []
        while cont and n<3:
            print("EM loop " + str(n) + '\r')
            log_likelihoods = np.zeros((len(data), len(self.pdfs)))
            for i, sample in enumerate(data):
                for j, pdf in enumerate(self.pdfs):
                    try:
                        log_likelihoods[i][j] = pdf.logpdf(sample)
                    except(np.linalg.LinAlgError):
                        print("occured loop "+str(n))

            Q = []
            for row in log_likelihoods:
                Q.append(row+np.log(self.weights) - logsumexp(row+np.log(self.weights)))
            Q = np.exp(np.array(Q))
            reg = np.sum(Q, axis=0)
            self.weights = reg / np.sum(reg)
            means = np.transpose(Q/reg)@data
            for i in range(len(self.pdfs)):
                cov = (Q[:,i] * np.transpose(data-self.pdfs[i].mean)) @ (data-self.pdfs[i].mean)
                cov /= reg[i]
                m_cont, m_change = net_change(means[i], self.pdfs[i].mean, threshold=.05)
                c_cont, c_change = net_change(cov, self.pdfs[i].covariance, threshold=1)
                # print("m_change: " + str(m_change))
                # print("c_change: " + str(c_change))
                cont = m_cont and c_cont
                self.pdfs[i].update_params(means[i], cov)
            mc.append(m_change)
            cc.append(c_change)
            n += 1
        return mc, cc

    def get_means(self):
        means = []
        for pdf in self.pdfs:
            means.append(pdf.mean)
        return np.array(means)

    def logpdf(self, sample):
        log_likelihoods = []
        for weight, pdf in zip(self.weights, self.pdfs):
            log_likelihoods.append(np.log(weight) + pdf.log_likelihood(sample))

        return logsumexp(np.array(log_likelihoods))

    def save(self, name):
        means = []
        cov = []
        for pdf in self.pdfs:
            means.append(pdf.mean)
            cov.append(pdf.covariance)
        means = np.array(means)
        cov = np.array(cov)
        np.save(os.path.join(get_p1_pickle_folder(), name+'mean'), means)
        np.save(os.path.join(get_p1_pickle_folder(), name+'cov'), cov)

    def load(self, name):
        means = np.load(os.path.join(get_p1_pickle_folder(), name+'mean'))
        cov = np.load(os.path.join(get_p1_pickle_folder(), name + 'cov'))
        for i, pdf in enumerate(self.pdfs):
            pdf.mean = means[i]
            pdf.cov = cov[i]


def run_mixture_gaussian():
    train_faces, train_background, test_faces, test_background = get_pickled_data()

    # f_model = MixtureGaussian(5, train_faces)
    # mc, cc = f_model.run_em(train_faces[:100])
    # plot_changes(mc, cc, title="MOG Changes per EM")
    # f_model.save("face_model")
    # return
    b_model = MixtureGaussian(5, train_background)
    b_model.run_em(train_background)

    # for i in range(5):
    #     imshow_sample(f_model.pdfs[i].mean)
    #     imshow_sample(f_model.pdfs[i].covariance[0])

    #     imshow_sample(b_model.pdfs[i].mean)
    #     imshow_sample(b_model.pdfs[i].covariance[0])
    #
    f_model = MixtureGaussian(5, train_faces)
    f_model.load('face_model')
    test_models(f_model, b_model, test_faces, test_background)