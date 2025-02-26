#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.autograd import grad, Variable
#from .linf_sgd import Linf_SGD

# Modified from PyTorch --> SGD optimizer
#import torch
from torch.optim.optimizer import Optimizer, required


class Linf_SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Linf_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Linf_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = torch.sign(p.grad.data)
                #d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss



# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def attack_Linf_PGD(input_v, ones, label_v, dis, Ld, steps, epsilon):
    dis.eval()
    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    optimizer = Linf_SGD([adverse_v], lr=0.0078)
    for _ in range(steps):
        optimizer.zero_grad()
        dis.zero_grad()
        d_bin, d_multi = dis(adverse_v)
        loss = -Ld(d_bin, ones, d_multi, label_v, lam=0.5)
        loss.backward()
        #print(loss.data[0])
        optimizer.step()
        diff = adverse_v.data - input_v.data
        diff.clamp_(-epsilon, epsilon)
        adverse_v.data.copy_((diff + input_v.data).clamp_(-1, 1))
    dis.train()
    dis.zero_grad()
    return adverse_v

# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def attack_Linf_PGD_bin(input_v, ones, dis, Ld, steps, epsilon):
    dis.eval()
    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    optimizer = Linf_SGD([adverse_v], lr=0.0078)
    for _ in range(steps):
        optimizer.zero_grad()
        dis.zero_grad()
        d_bin = dis(adverse_v)
        loss = -Ld(d_bin, ones)
        loss.backward()
        #print(loss.data[0])
        optimizer.step()
        diff = adverse_v.data - input_v.data
        diff.clamp_(-epsilon, epsilon)
        adverse_v.data.copy_((diff + input_v.data).clamp_(-1, 1))
    dis.train()
    dis.zero_grad()
    return adverse_v

# performs FGSM attack, and it is differentiable
# @input_v: make sure requires_grad = True
def attack_FGSM(input_v, ones, label_v, dis, Lg):
    dis.eval()
    d_bin, d_multi = dis(input_v)
    loss = -Lg(d_bin, ones, d_multi, label_v, lam=0.5)
    g = grad(loss, [input_v], create_graph=True)[0]
    return input_v - 0.005 * torch.sign(g)


# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def attack_label_Linf_PGD(input_v, label_v, dis, steps, epsilon):
    dis.eval()
    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    optimizer = Linf_SGD([adverse_v], lr=epsilon / 5)
    for _ in range(steps):
        optimizer.zero_grad()
        dis.zero_grad()
        _, d_multi = dis(adverse_v)
        loss = -F.cross_entropy(d_multi, label_v)
        loss.backward()
        #print(loss.data[0])
        optimizer.step()
        diff = adverse_v.data - input_v.data
        diff.clamp_(-epsilon, epsilon)
        adverse_v.data.copy_((diff + input_v.data).clamp_(-1, 1))
    dis.zero_grad()
    return adverse_v
