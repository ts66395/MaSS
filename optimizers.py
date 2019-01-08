from keras import backend as K
from keras.optimizers import Optimizer

# from backend_extra import scatter_update


class MaSS(Optimizer):
    """ Momentum-added Stochastic Solver.
    # Arguments
        lr: float >= 0. Learning rate.
        alpha: float >= 0 and < 1. 
        kappa_t: float > 0. Coefficient of the compensation term is computed by: lr/alpha/kappa_t.
    """

    def __init__(self, lr=0.1, alpha = 0.05, kappa_t = 12, 
                  **kwargs):
        super(MaSS, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
        self.kappa_t = K.variable(kappa_t, name='kappa_t')
        self.alpha = K.variable(alpha, name='alpha')


    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        kappa_t = self.kappa_t
        alpha = self.alpha
        shapes = [K.int_shape(p) for p in params]

        delta = lr/alpha/kappa_t

        #   Initialize weights/variables
        w = [K.variable(p) for p in params]
        v = [K.variable(p) for p in params]

        for ut, g, wt, vt in zip(params, grads, w, v):
            # Compute new weight values, following Eq.(14) of the MaSS paper: arXiv:1810.13395
            new_wt = ut - lr*g
            new_vt = (1-alpha)*vt + alpha*ut - delta*g
            new_ut = alpha/(1+alpha)*new_vt + 1/(1+alpha)*new_wt
 
	    # Update
            self.updates.append(K.update(vt, new_vt))
            self.updates.append(K.update(ut, new_ut))
            self.updates.append(K.update(wt, new_wt))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'kappa_t': float(K.get_value(self.kappa_t)),
                  'alpha': float(K.get_value(self.alpha))}
        base_config = super(MaSS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
