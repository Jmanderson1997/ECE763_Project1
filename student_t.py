import numpy as np
from math import factorial
from scipy.special import logsumexp, loggamma
from data import get_pickled_data
from roc import test_models, imshow_sample, net_change, plot_changes


class StudentT:
    def __init__(self, dof, samples, n_hidden=5):
        self.v = dof
        self.mean = np.mean(samples, axis=0)
        self.covariance = np.transpose(samples - self.mean) @ (samples - self.mean) / (len(samples)-1)

    def run_em(self, data):
        cont = True
        n = 0
        mc = []
        cc = []
        while(cont):
            print("EM loop "+str(n) +'\r')
            Eh = []
            for i, sample in enumerate(data):
                denomenator = self.v + np.transpose(sample-self.mean)@np.linalg.inv(self.covariance)@(sample-self.mean)
                Eh.append((self.v+1200)/denomenator)
            Eh = np.array(Eh)
            mean = Eh@data/np.sum(Eh)
            cov = (Eh*np.transpose(data-mean)) @ (data - mean) / np.sum(Eh)
            m_cont, m_change = net_change(mean, self.mean, threshold=.01)
            c_cont, c_change = net_change(cov, self.covariance, threshold=1)
            print("m_change: "+str(m_change))
            print("c_change: " + str(c_change))
            cont = m_cont and c_cont
            n += 1
            mc.append(m_change)
            cc.append(c_change)
            self.mean = mean
            self.covariance = cov

        return mc, cc

    def logpdf(self, data):
        x = loggamma((self.v+1200)/2) - (600*np.log(np.pi*self.v) + loggamma(self.v/2))
        _, y = np.linalg.slogdet(self.covariance)
        y *= -.5
        z = - (self.v+1200)/2 *logsumexp((1, (data-self.mean)@np.linalg.inv(self.covariance)@(x-self.mean)/2))
        return x + y +z


def run_t_dist():
    train_faces, train_background, test_faces, test_background = get_pickled_data()

    f_model = StudentT(10, train_faces)
    mf_change, cf_change = f_model.run_em(train_faces)

    b_model = StudentT(10, train_background)
    mb_channge, cb_change = b_model.run_em(train_background)

    plot_changes(mf_change, cf_change, 'Student T Change per EM Step')

    imshow_sample(f_model.mean)
    imshow_sample(f_model.covariance[0])

    imshow_sample(b_model.mean)
    imshow_sample(b_model.covariance[0])

    test_models(f_model, b_model, test_faces, test_background, title="Student T")





if __name__ == '__main__':
    run_t_dist()