import numpy as np
from data import get_pickled_data
from roc import test_models, imshow_sample, net_change, plot_changes


class FactorAnalyzer:

    def __init__(self, samples, n_h):
        self.mean = np.mean(samples, axis=0)
        self.covariance = np.diag(np.transpose(samples - self.mean) @ (samples - self.mean)) / (len(samples)-1)
        self.phi = np.random.normal(size=(n_h,1200))

    def run_em(self, data):
        cont = True
        mc =[]
        cc = []
        n = 0
        while cont and n<20:
            print("EM loop " + str(n) + '\r')
            phi_1 = np.zeros((len(self.phi),1200))
            phi_2 = np.zeros((len(self.phi),len(self.phi)))
            for sample in data:
                inv = np.linalg.inv(self.phi * self.covariance @ np.transpose(self.phi) + np.identity(len(self.phi)))
                eh = inv @ (self.phi*(1/self.covariance) @ (sample-self.mean))
                ehht = (inv + eh@np.transpose(eh))
                phi_1 += np.outer(eh, sample-self.mean)
                phi_2 += ehht

            phi = np.linalg.inv(phi_2) @phi_1
            sig = np.zeros(1200)

            for sample in data:
                inv = np.linalg.inv(self.phi * self.covariance @ np.transpose(phi) + np.identity(len(phi)))
                eh = inv @ (self.phi * (1 / self.covariance) @ (sample - self.mean))
                out = np.outer((sample-self.mean), (sample-self.mean))
                x = np.outer(eh@phi,(np.transpose(sample)-self.mean))
                sig += np.diag(out-x)

            sig /= len(data)

            m_cont, m_change = net_change(phi, self.phi, threshold=.05)
            c_cont, c_change = net_change(sig, self.covariance, threshold=1)
            print("m_change: " + str(m_change))
            print("c_change: " + str(c_change))
            cont = m_cont or c_cont
            n += 1
            mc.append(m_change)
            cc.append(c_change)

            self.phi = phi
            self.covariance = sig

        print("done")
        return mc, cc

    def logpdf(self, sample):
        cov = (np.transpose(self.phi)@self.phi + np.diag(self.covariance))
        sing_test = np.sum(np.diag(cov) == 0)
        if sing_test:
            cov += np.diag(np.ones(1200))
        x = -.5 * ((sample - self.mean) @ np.linalg.inv(cov) @ np.transpose(sample - self.mean))
        sign, y = np.linalg.slogdet(cov)
        y *= -.5
        z = -(1200 / 2) * np.log(2 * np.pi)
        return x + y + z


def run_factor_analyser():
    train_faces, train_background, test_faces, test_background = get_pickled_data()
    f_model = FactorAnalyzer(train_faces, n_h=20)
    phi_c, cc = f_model.run_em(train_faces)

    b_model = FactorAnalyzer(train_background, n_h=20)
    b_model.run_em(train_background)

    imshow_sample(b_model.mean)
    imshow_sample((np.transpose(b_model.phi)@b_model.phi + np.diag(b_model.covariance))[0])

    plot_changes(phi_c, cc, title="MOG Changes per EM")

    test_models(f_model, b_model, test_faces, test_background, "Factor Analyzer")


if __name__ == '__main__':
    run_factor_analyser()
