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

# It is useful (although not required) to define a little
# dictionary stores coordinate name & size information
# Ideally we can follow best use practices where we avoid
# repeatedly re-defining coordinates, but I haven't quite
# figured out those patterns yet, so for now I just kind of
# refer to "x" and "t" repeatedly throught the code.
coords = {"x": (1,), "t": (1,)}

# Define coordinate domain
x_min = -1.0
x_max = 1.0
t_min = 0.0
t_max = 1.0

# The very first thing that needs to be done is to define the input and output remaps
# in the future I will define a bunch of common remaps in pinn_kit, but for now everything
# is manual.
# Note: this would be the ideal place to do some de-scaling, but right now I haven't bothered.
in_map = InputRemap(coords, lambda crds: (crds["t"], crds["x"]))
out_map = OutputRemap(coords, (torch.Size((1,)),), lambda crds, vrs: {"u": vrs[0]})

# Next, we use the above mappings to define a solution, which is a trainable function between
# the named coordinates of its domain and the named variables of its range.
solution = ClassicPINN(in_map, out_map)

# For convenience, we should define sympy symbols for the coordinates & variables.
x_sym, t_sym, u_sym = sp.symbols("x t u")

# Now we create the equation that the PINN will try to solve
burgers_eq = Equation(
    solution, sp.Derivative(u_sym, t_sym) + u_sym * sp.Derivative(u_sym, x_sym) - (0.01 / 3.14) * sp.Derivative(u_sym, x_sym, 2)
)

# Next, a helper function to generate initial data
def u_0(x):
    return -torch.sin(np.pi*((2.*(x-x_min)/(x_max-x_min))-1.))

# To keep track of how (badly) things are going, a convenience function to plot the solution at various times
def plot_current_solution():
    x_vals = torch.linspace(x_min, x_max, 100).reshape(-1, 1)
    ones = torch.ones_like(x_vals)
    plt.plot(x_vals, u_0(x_vals), label=r"$u_0$")
    for t in np.linspace(0, 1, 10):
        u_vals = solution({"t" : t*ones, "x" : x_vals})["u"]
        plt.plot(x_vals, u_vals.detach().numpy(), label=f"NN, $t={t}$")
    plt.legend()
    plt.show()
    plt.clf()
    plt.close()

# Convenience functions to generate points on the initial time, the boundaries, and the colocation points of the bulk.
def initial_points(N_x):
    x_samp = torch.linspace(x_min, x_max, N_x).reshape(-1, 1)
    return {"x" : x_samp, "t" : torch.zeros_like(x_samp)}

def boundary_points(N_t):
    t_samp = torch.linspace(t_min, t_max, N_t).reshape(-1, 1)
    left_bound = {"x" : torch.ones_like(t_samp)*x_min, "t" : t_samp}
    right_bound = {"x" : torch.ones_like(t_samp)*x_max, "t" : t_samp}
    return left_bound, right_bound

def bulk_points(N_bulk, t_max):
    lower_bound = np.array([t_min, x_min], dtype='float32')
    upper_bound = np.array([t_max, x_max], dtype='float32')
    bulk_samples = torch.from_numpy(lower_bound + (upper_bound - lower_bound)*lhs(2, N_bulk).astype('float32'))
    return {"t" : bulk_samples[:, 0].reshape(-1, 1), "x" : bulk_samples[:, 1].reshape(-1, 1)}

# pinn_kit doesn't have any standard initial & boundary loss functions yet. I think that the best way to do things is to make
# initial and boundary conditions automatic through the mappings anyway.
def initial_loss(sol, initial_points):
    return torch.mean(torch.square(sol(initial_points)["u"] - u_0(initial_points["x"])))
    
def boundary_loss(sol, left_bound, right_bound):
    #return torch.mean(torch.square(sol(left_bound)["u"] - sol(right_bound)["u"]))
    return torch.mean(torch.square(sol(left_bound)["u"]) + torch.square(sol(right_bound)["u"]))

# Now we define an optimizer...
optimizer = torch.optim.AdamW(solution.parameters(), lr = 3e-4)

# Create training data...
initial = initial_points(100)
left, right = boundary_points(100)

# And start training!

# Why not just add a .train() method to the solution object? Well I could do that, but my thinking is that
# .train() sort of implies we know how to train PINNs effectively, and we really don't. Case in point:
# the procedure below completely fails to produce a good solution to the burger's equation. Probably I
# haven't run things long enough, but still, there are some serious hyperparameter tuning problems here,
# and just re-creating results of other papers has been tough. There are a plethora of training techniques
# and they involve a lot of non-standard things (variational losses instead of co-location points, adaptive
# re-sampling, constrained-optimization-style co-location point re-weighting, just to name a few).
# Thinking about these issues, I decided it was premature to write a default .train() procedure in pinn_kit
# and instead each script / notebook we write should explicitly impliment a training procedure so that we
# can focus on identifying good strategies for our problems of interest!

start_time = time.time()
for steps in range(1, 1500):
    loss = 100*initial_loss(solution, initial) + 10*boundary_loss(solution, left, right)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print(f"Loss {loss.item()}")

plot_current_solution()

bulk = bulk_points(1000, t_max/2.)
for steps in range(1, 1000):
    loss = 10*initial_loss(solution, initial) + 10*boundary_loss(solution, left, right) + ColocationLoss(burgers_eq.f, bulk, torch.ones_like(bulk["x"]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print(f"Loss {loss.item()}")

plot_current_solution()

bulk = bulk_points(1000, t_max)
for steps in range(1, 1000):
    loss = 100*initial_loss(solution, initial) + 10*boundary_loss(solution, left, right) + ColocationLoss(burgers_eq.f, bulk, torch.ones_like(bulk["x"]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print(f"Loss {loss.item()}")

plot_current_solution()


bulk = bulk_points(5000, t_max)
for steps in range(1, 500):
    loss = 100*initial_loss(solution, initial) + 10*boundary_loss(solution, left, right) + ColocationLoss(burgers_eq.f, bulk, torch.ones_like(bulk["x"]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if steps % 10 == 0:
        print(f"Loss {loss.item()}")


end_time = time.time()

print(end_time - start_time)

plot_current_solution()
