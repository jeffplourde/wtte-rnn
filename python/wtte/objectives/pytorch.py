""" Objective functions for PyTorch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


def loglik_continuous(a, b, y_, u_):
    """ Returns element-wise Weibull censored log-likelihood.

    Continuous weibull log-likelihood. loss=-loglikelihood.
    All input values must be of same type and shape.

    :param a:alpha. Positive nonzero `Tensor`.
    :type a: `FloatTensor` or `DoubleTensor`.
    :param b:beta.  Positive nonzero `Tensor`.
    :type b: `FloatTensor` or `DoubleTensor`.
    :param y_: time to event. Positive nonzero `Tensor`
    :type y_: `FloatTensor` or `DoubleTensor`.
    :param u_: indicator. 0.0 if right censored, 1.0 if uncensored `Tensor`
    :type u_: `FloatTensor` or `DoubleTensor`.
    :return: A `Tensor` of log-likelihoods of same shape as a, b, y_, u_
    """
    ya = (y_ + 1e-35) / a  # Small optimization y/a

    loglik = (u_ * (torch.log(b) + (b * torch.log(ya)))) - torch.pow(ya, b)

    return(loglik)


def loglik_discrete(a, b, y_, u_):
    """Returns element-wise Weibull censored discrete log-likelihood.

    Unit-discretized weibull log-likelihood. loss=-loglikelihood.

    .. note::
        All input values must be of same type and shape.

    :param a:alpha. Positive nonzero `Tensor`.
    :type a: `FloatTensor` or `DoubleTensor`.
    :param b:beta.  Positive nonzero `Tensor`.
    :type b: `FloatTensor` or `DoubleTensor`.
    :param y_: time to event. Positive nonzero `Tensor` 
    :type y_: `FloatTensor` or `DoubleTensor`.
    :param u_: indicator. 0.0 if right censored, 1.0 if uncensored `Tensor`
    :type u_: `FloatTensor` or `DoubleTensor`.
    :return: A `Tensor` of log-likelihoods of same shape as a, b, y_, u_.
    """
    hazard0 = torch.pow(((y_ + 1e-35) / a), b)  # 1e-9 safe, really
    hazard1 = torch.pow(((y_ + 1.0) / a), b)
    loglik = u_ * (torch.log(
        torch.exp(hazard1 - hazard0) - 1.0)) - hazard1

    return(loglik)


def betapenalty(b, location=10.0, growth=20.0):
    """Returns a positive penalty term exploding when beta approaches location.

    Adding this term to the loss may prevent overfitting and numerical instability
    of large values of beta (overconfidence). Remember that loss = -loglik+penalty

    :param b:beta.  Positive nonzero `Tensor`.
    :type b: `FloatTensor` or `DoubleTensor`.
    :param output_collection:name of the collection to collect result of this op.
    :type output_collection: Tuple of Strings.
    :param String name: name of the operation.
    :return:  A positive `Tensor` of same shape as `b` being a penalty term.
    """
    scale = growth / location
    penalty = torch.exp(scale * (b - location))

    return(penalty)
