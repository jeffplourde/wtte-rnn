from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest

import torch
import numpy as np

from torch.autograd import Variable

from wtte.objectives.pytorch import loglik_continuous, loglik_discrete
from wtte.data_generators import generate_weibull

# SANITY CHECK: Use pure Weibull data censored at C(ensoring point).
# Should converge to the generating A(alpha) and B(eta) for each timestep

n_samples = 1000
n_features = 1
real_a = 3.
real_b = 2.
censoring_point = real_a * 2


def tf_loglik_runner(loglik_fun, discrete_time):
    np.random.seed(1)
    torch.manual_seed(1)

    a = torch.autograd.Variable(torch.exp(torch.ones(1)), requires_grad=True)
    b = torch.autograd.Variable(torch.exp(torch.ones(1)), requires_grad=True)

    # testing part:
    _, tte_censored, u_train = generate_weibull(
        A=real_a,
        B=real_b,
        C=censoring_point,  # <np.inf -> impose censoring
        shape=[n_samples, n_features],
        discrete_time=discrete_time)

    tte_censored = torch.from_numpy(tte_censored).float()
    u_train = torch.from_numpy(u_train).float()
    
    y = Variable(tte_censored, requires_grad=False)
    u = Variable(u_train, requires_grad=False)

    learning_rate = 0.008
    for step in range(1000):
        loglik = loglik_fun(a,b,y,u)
        loss = -torch.mean(loglik)
        loss.backward()
        
        a.data -= learning_rate * a.grad.data
        b.data -= learning_rate * b.grad.data

        a.grad.data.zero_()
        b.grad.data.zero_()

        if step % 100 == 0:
            print(step, loss.data[0], a.data[0], b.data[0])

    print(torch.abs(real_a - a.data)[0], torch.abs(real_b - b.data)[0])
    assert torch.abs(real_a - a.data)[0] < 0.05, 'alpha not converged'
    assert torch.abs(real_b - b.data)[0] < 0.05, 'beta not converged'

def test_loglik_continuous():
    tf_loglik_runner(loglik_continuous, discrete_time=False)

def test_loglik_discrete():
    tf_loglik_runner(loglik_discrete, discrete_time=True)
