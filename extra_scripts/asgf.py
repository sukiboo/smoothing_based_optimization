import time
import json
import numpy as np


class ASGF:
    """Adaptive Stochastic Gradient-Free algorithm"""

    def __init__(self, s0=None, s_rate=.9, s_min=1e-03, s_max=1e+03,\
                m_min=5, m_max=25, qtol=.1, quad_rule='gh',\
                L_lmb=.9, lr_min=1e-03, lr_max=1e+03,\
                A_grad=.1, A_dec=.95, A_inc=1.02,\
                B_grad=.9, B_dec=.98, B_inc=1.01,\
                restart=False, num_res=2, res_mlt=10, res_div=10, fun_req=-np.inf,\
                xtol=1e-06, maxiter=1000):
        '''initialize class variables'''
        self.s0 = s0
        self.s_rate = s_rate
        self._s_min, self._s_max = s_min, s_max
        self.m_min, self.m_max = m_min, m_max
        self.qtol, self.quad_rule = qtol, quad_rule
        self.L_avg, self.L_lmb = 0, L_lmb
        self.lr_min, self.lr_max = lr_min, lr_max
        self.A_grad, self.B_grad = A_grad, B_grad
        self.A_dec, self.A_inc = A_dec, A_inc
        self.B_dec, self.B_inc = B_dec, B_inc
        self.restart = restart
        self.num_res = num_res
        self.res_mlt, self.res_div = res_mlt, res_div
        self.fun_req = fun_req
        self.update_rule = update_rule
        self.xtol = xtol
        self.maxiter = maxiter
        self.record_parameters()
        if self.quad_rule == 'gh':
            self.quad = self.gh_quad

    def optimization_init(self, fun, x0, subname):
        '''initialize optimization variables'''
        self.fun = fun
        self.x = np.array(x0)
        self.dim = self.x.size
        self.dx = np.zeros(self.dim)
        self.df = np.zeros(self.dim)
        self.lr = 0
        self.f = self.fun(self.x)
        self.itr, self.feval = 0, 1
        self.x_min, self.f_min = self.x, self.f
        self.reset_params()
        self.s_min = self.s / 1000 if self._s_min is None else self._s_min
        self.s_max = self.s * 1000 if self._s_max is None else self._s_max
        self.s_res = self.s
        self.converged = False
        self.subname = '' if subname is None else '_' + str(subname)
        self.record_iteration(log_mode='w')

    def generate_directions(self, vec=None):
        '''generate search directions'''
        if vec is None:
            self.u = np.random.randn(self.dim, self.dim)
        else:
            self.u = np.concatenate((vec.reshape((-1,1)),\
                np.random.randn(self.dim, self.dim-1)), axis=1)
        self.u /= np.linalg.norm(self.u, axis=0)
        self.u = np.linalg.qr(self.u)[0].T

    def reset_params(self):
        '''set parameters to their initial values'''
        self.generate_directions()
        self.s = np.sqrt(self.dim) if self.s0 is None else self.s0
        self.s_status = '*'
        self.L, self.A, self.B = 0, self.A_grad, self.B_grad

    def minimize(self, fun, x0, subname=None):
        '''iteratively update minimizer'''
        self.optimization_init(fun, x0, subname)
        while (self.itr < self.maxiter) and not self.converged:
            # compute gradient and update minimizer
            self.compute_df()
            self.update_x()
            self.save_state()
            # update variables
            self.itr += 1
            self.update_parameters()
            self.record_iteration()

    def gh_quad(self, g, s, g0, mode='adaptive'):
        '''compute derivative via adaptive Gauss-Hermite quadrature'''
        dg_quad = np.array([np.inf])
        g_vals, p_vals, feval_gh = np.array([]), np.array([]), 0
        num_pts = range(max(3, self.m_min-2), self.m_max+1, 2)\
            if mode == 'adaptive' else [self.m_min]
        # iteratively estimate smoothed derivative
        for m in num_pts:
            p, w = np.polynomial.hermite.hermgauss(m)
            g_val = np.array([g(p_i * s) for p_i in p[p != 0]])
            feval_gh += m - 1
            g_val = np.insert(g_val, *np.where(p == 0), g0)
            g_vals = np.append(g_vals, g_val)
            p_vals = np.append(p_vals, p)
            dg_quad = np.append(dg_quad, np.sum(w * p * g_val) / (s * np.sqrt(np.pi) / 2))
            # compute relative difference for gradient estimate
            qdelta = np.abs(dg_quad[:-1] - dg_quad[-1]) / (np.abs(dg_quad[-1]) + 1e-06)
            if np.amin(qdelta) < self.qtol:
                break
        p, p_ind = np.unique(p_vals, return_index=True)
        g_val = g_vals[p_ind]
        # estimate local Lipschitz constant
        L = np.amax(np.abs(g_val[1:] - g_val[:-1]) / (p[1:] - p[:-1]) / s)
        return dg_quad[-1], feval_gh, L

    def compute_df(self):
        '''estimate gradient from directional derivatives'''
        mode = ['adaptive' if i==0 else None for i in range(self.dim)]
        if self.parallel:
            self.dg, feval, self.L = zip(*ray.get([self.quad_parallel.remote(\
                lambda t : self.fun(self.x + t * self.u[d]), self.s, self.f, mode[d])\
                for d in range(self.dim)]))
        else:
            self.dg, self.L = np.zeros(self.dim), np.zeros(self.dim)
            feval = np.zeros(self.dim, dtype=int)
            for d in range(self.dim):
                self.dg[d], feval[d], self.L[d] = self.quad(\
                    lambda t : self.fun(self.x + t * self.u[d]), self.s, self.f, mode[d])
        self.feval += np.sum(feval)
        self.df = np.matmul(self.dg, self.u)

    def update_x(self):
        '''update minimizer'''
        # select learning rate
        self.L_avg = self.L[0] if self.L_avg==0\
            else (1 - self.L_lmb) * self.L[0] + self.L_lmb * self.L_avg
        self.lr = np.clip(self.s / self.L_avg, self.lr_min, self.lr_max)
        if self.update_rule == 'adam':
            # adam update
            self.mt = self.adam_beta[0] * self.mt + (1 - self.adam_beta[0]) * self.df
            self.vt = self.adam_beta[1] * self.vt + (1 - self.adam_beta[1]) * self.df**2
            mh = self.mt / (1 - self.adam_beta[0]**(self.itr + 1))
            vh = self.vt / (1 - self.adam_beta[1]**(self.itr + 1))
            self.dx = self.lr * mh / (np.sqrt(vh) + self.adam_eps)
        else:
            # gradient descent
            self.dx = self.lr * self.df
        # update minimizer
        self.x -= self.dx
        self.f = self.fun(self.x)
        self.feval += 1

    def save_state(self):
        '''save the best state'''
        if self.f < self.f_min:
            self.x_min = self.x.copy()
            self.f_min = self.f
            self.s_res = self.s

    def restart_state(self):
        '''restart from the best state'''
        self.x = self.x_min.copy()
        self.f = self.f_min
        self.s = self.s_res

    def update_parameters(self):
        '''update hyperparameters'''
        self.converged = np.amax(np.abs(self.dx)) < self.xtol
        flag_conv = self.s < self.s_min * self.res_mlt
        flag_div = self.s > self.s_max / self.res_div
        if self.restart and ((flag_conv and self.num_reset > 0) or flag_div):
            # reset parameters / restart state
            self.reset_params()
            self.num_res -= 1
            if self.num_res == -1 or flag_div:
                self.restart_state()
            if self.verbose > 1:
                print('iteration {:d}: resetting the parameters'.format(self.itr+1))
        else:
            # update directions and adjust smoothing parameter
            self.generate_directions(self.df)
            s_norm = np.amax(np.abs(self.dg) / self.L)
            if s_norm < self.A:
                self.s *= self.s_rate
                self.A *= self.A_dec
                self.s_status = '-'
            elif s_norm > self.B:
                self.s /= self.s_rate
                self.B *= self.B_inc
                self.s_status = '+'
            else:
                self.A *= self.A_inc
                self.B *= self.B_dec
                self.s_status = '='
            self.s = np.clip(self.s, self.s_min, self.s_max)

    def record_parameters(self):
        '''record hyperparameters'''
        os.makedirs('./logs/', exist_ok=True)
        json.dump(self.__dict__, open('./logs/' + self.log_name + '.json', 'w'))

    def record_iteration(self, log_mode='a'):
        '''record current variables'''
        with open('./logs/' + self.log_name + self.subname + '.csv', log_mode) as logfile:
            if log_mode == 'w':
                log_func = lambda x: x.replace('self.', '')
                logfile.write(','.join(map(log_func, self.log_vars)) + '\n')
            log_func = lambda x: self.log_vars[x].format(eval(x))
            logfile.write(','.join(map(log_func, self.log_vars)) + '\n')
        self.display_variables()

    def display_variables(self):
        '''display current variables'''
        if self.verbose > 0:
            print('iteration {:d}:  f = {:.2e},  lr = {:.2e},  s = {:.2e} ({:s})'.\
                format(self.itr, self.f, self.lr, self.s, self.s_status))
        if self.verbose > 1:
            print('  df_2 = {:.2e},  dx_2 = {:.2e},  dx_inf = {:.2e}'.\
                format(np.linalg.norm(self.df), np.linalg.norm(self.dx), np.amax(np.abs(self.dx))))
        if self.verbose > 2:
            print('   x =', self.x[:10])
            print('  df =', self.df[:10])


if __name__ == '__main__':

    # problem setup
    fun_name = 'ackley'
    fun_dim = 10
    fun, x_min, x_dom = target_function(fun_name, fun_dim)
    x0 = initial_guess(x_dom)

    # run asgf optimization
    asgf = ASGF(log_name=log_name, verbose=1, parallel=False)
    asgf.minimize(fun, x0, subname='test')
    x_opt = asgf.x_min
    itr_opt = asgf.itr
    feval_opt = asgf.feval

    # report results
    f_delta = np.abs((fun(x_opt) - fun(x_min)) / (fun(x0) - fun(x_min)))
    print('\nasgf-optimization terminated after {:d} iterations and {:d} evaluations: '\
        'f_min = {:.2e} / {:.2e}'.format(itr_opt, feval_opt, fun(x_opt), f_delta))


