#!/usr/bin/env python
import torch
from torch import nn
from pyDOE import lhs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

import sympy as sp

from pinn_kit import *

coords = {"x": (1,), "t": (1,)}

in_map = InputRemap(coords, lambda crds: (crds["t"], crds["x"]))

out_map = OutputRemap(coords, (torch.Size((1,)),), lambda crds, vrs: {"u": vrs[0]})

x_min = -1.0
x_max = 1.0

t_min = 0.0
t_max = 1.0

X = torch.linspace(x_min, x_max, 100).reshape(-1, 1)
T = torch.linspace(t_min, t_max, 100).reshape(-1, 1)

coords_in = {"x": X, "t": T}

model = ClassicPINN(in_map, out_map)

print(model)

print(model(coords_in))

x_sym, t_sym, u_sym = sp.symbols("x t u")

algebraic_test = Equation(model, u_sym - sp.sin(x_sym) * sp.cos(t_sym))

print(algebraic_test)

print(algebraic_test.symbolic_expression)

print(algebraic_test.f(coords_in))

diffeq_test = Equation(
    model, sp.Derivative(u_sym, t_sym) + u_sym * sp.Derivative(u_sym, x_sym) - (0.01 / 3.14) * sp.Derivative(u_sym, x_sym, 2)
)

print(diffeq_test)

print(diffeq_test.symbolic_expression)

print(diffeq_test.f(coords_in))

print(diffeq_test._eq_eval.__doc__)

print(ColocationLoss(diffeq_test.f, coords_in, torch.ones_like(coords_in["x"])))

#
# t_min = 0.
# t_max = 1.
# x_min = -1.
# x_max = 1.
#
# def u_0(x):
#    return -torch.sin(np.pi*((2.*(x-x_min)/(x_max-x_min))-1.))
#
# def initial_points(N_x):
#    x_samp = torch.linspace(x_min, x_max, N_x).reshape(-1, 1)
#    return torch.cat((torch.zeros_like(x_samp), x_samp), 1)
#
# def boundary_points(N_t):
#    t_samp = torch.linspace(t_min, t_max, N_t).reshape(-1, 1)
#    left_bound = torch.cat((t_samp, torch.ones_like(t_samp)*x_min), 1)
#    right_bound = torch.cat((t_samp, torch.ones_like(t_samp)*x_max), 1)
#    return left_bound, right_bound
#
# def bulk_points(N_bulk):
#    lower_bound = np.array([t_min, x_min], dtype='float32')
#    upper_bound = np.array([t_max, x_max], dtype='float32')
#    bulk_samples = lower_bound + (upper_bound - lower_bound)*lhs(2, N_bulk).astype('float32')
#    return torch.from_numpy(bulk_samples)
#
# def initial_loss(initial_points):
#    x_points = torch.split(initial_points, 1, dim=1)[1]
#    return torch.mean(torch.square(u(initial_points) - u_0(x_points)))
#
# def boundary_loss(left_bound, right_bound):
#    #return torch.mean(torch.square(u(left_bound) - u(right_bound)))
#    return torch.mean(torch.square(u(left_bound)) + torch.square(u(right_bound)))
#
# def bulk_loss(bulk_samples):
#    return torch.mean(torch.square(f(*torch.split(bulk_samples, 1, dim=1))))
#
# optimizer = torch.optim.AdamW(u.parameters(), lr = 3e-4)
#
## For now, every call to PINN_Loss() creates a new batch, so there's no need
## to talk about epochs vs. batches. Might not be optimal, I'm not sure.
#
#
# initial = initial_points(100)
# left, right = boundary_points(100)
# bulk = bulk_points(10000)
#
# start_time = time.time()
# for steps in range(1, 1500):
#    loss = 100*initial_loss(initial) + 10*boundary_loss(left, right)
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#    #if steps % 10 == 0:
#    #    print(f"Loss {loss.item()}")
#
# for steps in range(1, 100):
#    loss = 100*initial_loss(initial) + 10*boundary_loss(left, right) + bulk_loss(bulk)
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#    #if steps % 10 == 0:
#    #    print(f"Loss {loss.item()}")
#
# end_time = time.time()
#
# print(end_time - start_time)
#
# x_vals = torch.linspace(x_min, x_max, 100).reshape(-1, 1)
# t_vals_1 = torch.zeros_like(x_vals)
# plot_points_1 = torch.cat((t_vals_1, x_vals), 1)
# u_vals_1 = u(plot_points_1)
# plt.plot(x_vals, u_0(x_vals), label=r"$u_0$")
# plt.plot(x_vals, u_vals_1.detach().numpy(), label=r"NN, $t=0$")
# t_vals_2 = torch.ones_like(x_vals)*0.4
# plot_points_2 = torch.cat((t_vals_2, x_vals), 1)
# u_vals_2 = u(plot_points_2)
# plt.plot(x_vals, u_vals_2.detach().numpy(), label=r"NN, $t=0.4$")
# t_vals_3 = torch.ones_like(x_vals)*0.7
# plot_points_3 = torch.cat((t_vals_3, x_vals), 1)
# u_vals_3 = u(plot_points_3)
# plt.plot(x_vals, u_vals_3.detach().numpy(), label=r"NN, $t=0.7$")
# plt.legend()
# plt.show()
