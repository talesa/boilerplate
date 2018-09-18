import subprocess

from numbers import Number
import torch
import math

from torch import tensor


__all__ = ['git_revision', 'log_sum_exp']


def git_revision():
    return subprocess.check_output("git rev-parse --short HEAD".split()).strip().decode("utf-8")


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    Courtesy of JW
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
