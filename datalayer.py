import numpy as np
import sklearn.neighbors as nn
import tensorflow as tf

# NOTE: all the shape of data is (N) * H * W * C

binsfile = './resources/pts_in_hull.npy'
priorfile = './resources/prior_probs.npy'

class Quantilization():
    def __init__(self, binsfile=binsfile):
        self.NN = 10
        self.sigma = 5
        self.cc = np.float32(np.load(binsfile) / 220.0 + 0.5)
        self.K = self.cc.shape[0]
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='ball_tree').fit(self.cc)


    def __call__(self, abdata):
        '''
        input: h * w * 2
        return: h * w * 313
        '''
        h, w, c= abdata.shape
        flt_ab = np.reshape(abdata, [h * w, c])
        (dists, inds) = self.nbrs.kneighbors(flt_ab)

        wts = np.exp(-dists**2/(2 * self.sigma**2))
        wts = wts / np.sum(wts, axis=1)[:, np.newaxis]

        p_inds = np.arange(h * w, dtype='int')[:, np.newaxis]

        result = np.zeros((h * w, self.K), np.float32)
        result[p_inds, inds] = wts.astype(np.float32)
        result = np.reshape(result, (h, w, self.K))
        return result


class ClassRebalanceWeight():
    def __init__(self, priorfile=priorfile):
        self.gamma = 0.5
        self.alpha = 1.0
        # empirical prior probability
        self.prior_probs = np.load(priorfile)
        # uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)
        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs
        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # re-normalize
        self.prior_factor = np.float32(self.prior_factor)
        # implied empirical prior
        # self.implied_prior = self.prior_probs*self.prior_factor
        # self.implied_prior = self.implied_prior/np.sum(self.implied_prior) # re-normalize

    def __call__(self, quan_abdata):
        '''
        input: h * w * 313
        return: h * w * 1
        '''
        data_ab_maxind = np.argmax(quan_abdata, axis=-1)
        corr_factor = self.prior_factor[data_ab_maxind]
        corr_factor = np.expand_dims(corr_factor, axis=-1)
        return corr_factor


def decode_to_ab(quan_ab, binsfile=binsfile, namescope='decode_to_ab'):
    with tf.variable_scope(namescope):
        cc = np.load(binsfile) / 220.0 + 0.5
        cc = tf.constant(cc, dtype=np.float32)
        res = tf.tensordot(quan_ab, cc, [[-1], [0]])
    return res

# instantiation
quantilization = Quantilization()
rebalanceweight = ClassRebalanceWeight() 
