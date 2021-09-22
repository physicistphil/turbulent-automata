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
    model, sp.Derivative(u_sym, t_sym) + u_sym * sp.Derivative(u_sym, x_sym) - (sp.S(1)/31) * sp.Derivative(u_sym, x_sym, 2)
)

print(diffeq_test)

print(diffeq_test.symbolic_expression)

print(diffeq_test.f(coords_in))

print(ColocationLoss(diffeq_test.f, coords_in, torch.ones_like(coords_in["x"])))

expr1 = Expression({**coords, "u": (1,)}, u_sym - sp.sin(x_sym) * sp.cos(t_sym))

print(expr1)

print(expr1.symbols)
print(expr1.symbolic_expression)
print(expr1({**coords_in, **model(coords_in)}))

expr2 = Expression({**coords, "u": (1,)}, diffeq_test.symbolic_expression)

print(expr2)

print(expr2.symbols)
print(expr2.symbolic_expression)
print(expr2({**coords_in, **model(coords_in)}))

print(expr2({**coords_in, **model(coords_in)}) == diffeq_test.f(coords_in))

expr3 = Expression(coords, diffeq_test.f)

print(expr3)

print(expr3.symbols)
print(expr3.symbolic_expression)
print(expr3(coords_in))

print(expr2({**coords_in, **model(coords_in)}) == expr3(coords_in))


beta_sym, U_sym, T_sym = sp.symbols(r"\beta U T")
sub_exp = Expression({**coords, U_sym.name: (1,)}, (sp.S(1)/31) * sp.Derivative(U_sym, x_sym, 2))
sub_exp_u = Expression({u_sym.name: (1,)}, lambda syms: syms[u_sym.name], (1,))
sub_exp_t = Expression({t_sym.name: (1,)}, lambda syms: syms[t_sym.name], (1,))
sub_exp_test = Equation(model, sp.Derivative(U_sym, T_sym) + U_sym * sp.Derivative(U_sym, x_sym) - beta_sym, [(beta_sym.name, sub_exp), (U_sym.name, sub_exp_u), (T_sym.name, sub_exp_t)])

print(sub_exp_test)
print(sub_exp_test.f(coords_in))
print(sub_exp_test.symbolic_expression)
print(diffeq_test.symbolic_expression)

print(sub_exp_test.f(coords_in) - diffeq_test.f(coords_in))
