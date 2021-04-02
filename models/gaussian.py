import numpy as np
import os
from dataset.data import get_pickled_data
from scipy.special import logsumexp
from utils.roc import test_models, imshow_sample, net_change


class Gaussian:
    def __init__(self, mean=None, covariance=None, samples=None):
        if (mean is None or covariance is None) and samples is None:
            raise RuntimeError("Please provide samples or direct mean/covariance arrays")
        elif samples is not None:
            self.mean = np.mean(samples, axis=0)
            self.covariance = np.transpose(samples-self.mean)@(samples-self.mean)/(len(samples)-1)
        else:
            self.mean = mean
            self.covariance = covariance

    def logpdf(self, sample):
        sing_test = np.sum(np.diag(self.covariance) == 0)
        if sing_test:
            self.covariance = self.covariance + np.diag(np.ones(1200))
        x = -.5*((sample - self.mean) @ np.linalg.inv(self.covariance) @ np.transpose(sample - self.mean))
        sign, y = np.linalg.slogdet(self.covariance)
        y *= -.5
        z = -(1200/2)*np.log(2*np.pi)
        return x+y+z

    def update_params(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance


def single_gaussian():
    train_faces, train_background, test_faces, test_background = get_pickled_data()

    f_model = Gaussian(samples=train_faces)
    b_model = Gaussian(samples=train_background)

    imshow_sample(f_model.mean)
    imshow_sample(f_model.covariance[0])

    imshow_sample(b_model.mean)
    imshow_sample(b_model.covariance[0])

    # f_model_t = multivariate_normal(mean=f_model.mean, cov=f_model.covariance)
    # b_model_t = multivariate_normal(mean=b_model.mean, cov=b_model.covariance)

    # real = multivariate_normal.logpdf(test_faces[1], mean=f_model.mean, cov=f_model.covariance)

    test_models(f_model, b_model, test_faces, test_background, "Single Gaussian")




if __name__ == '__main__':
    single_gaussian()