#!/usr/bin/env python
import torch
from torch import nn
#from torch.nn import functional as F #Not being used?
from pyDOE import lhs
import numpy as np
import sympy as sp

"""PINN Kit contains a suite of helpful classes and functions for building & training
models in the PINN method to solve differential equations. The goal is to save time
and effort, and make rapid iteration & experimentation possible. In particular, the
focus is on the GR initial data problem, which can involve simple laplace-like
equations in 3D, but may involve much more complicated coupled PDE systems in 3D or
higher, over multiple domains with custom junction and boundary conditions.
Because of this potential complexity, we have opted to implement our own modest
framework, rather than relying on existing ones like neurodiffeq.

The contents of the kit come in four categories:

1. Model-building helper classes & functions
These allow models to be defined quickly & simply, and relatively independently from
the equations the models aim to solve. In particular, machinery exists to allow the
number of input variables to be decided at instantiation, along with how these inputs
will be processed (allowing some models to create parallel 'trunks' that process some
variables seperately, for example).

2. Core PINN tools
The central idea of the PINN technique is to train with a loss that penalizes failure
to exactly satisfy a differential equation. This module provides a quick way of
specifying equations via sympy expressions, and creating a penalty function f()
that can be evaluated at co-location points throughout the domain as part of the loss
function during training. In addition, functions for generating random samples and
good random model initialization are provided.

3. Uniqe Loss Functions & Training Algorithms
PINNs represent a new problem category in ML, with properties that are not present
in many classification / prediction problems commonly studied. It is unlikely that
the training techniques that are optimized for traditional ML will be optimal for
PINNs, and it is therefore necessary to play around with custom techniques inspired
by the unique challenges PINNs face. Implementations of some custom optimization
methods and unique loss functions are kept here.

4. Implementations of Unique Model Architectures
To be efficient enough to improve upon traditional PDE solution methods, PINNs need
to use different architectures biased toward their problem type. Many archetectures
have been proposed, and this module contains implementations of some that we've tried."""

class InputRemap():
    """Defines a map between input coordinates and the model input.
    This is useful for inducing a natural topology that can allow automatic
    boundary condition solution.

    Parameters:
    coords_in : dict of coordinate names and tensor sizes
    mapping_function : A function that takes an input coordinate dictionary and outputs a tuple of tensors
    coords_out : tuple with the output tensor sizes"""
    def __init__(self, coords_in, mapping_function):
        self.coords_in = coords_in
        self.mapping_function = mapping_function
        test_input = {}
        for name, size in coords_in.items():
            test_input[name] = (torch.zeros(size))[None, ...]
        test_output = mapping_function(test_input)
        self.coords_out = tuple(out_coord[0, ...].size() for out_coord in test_output)

    def __call__(coords):
        return self.mapping_function(coords)

class OutputRemap():
    """Defines a coordinate-dependent map between model output and the solution space.
    This is useful for imposing a solution ansatz that automatically satisfies boundary
    or junction conditions.
    
    Parameters:
    coords_in : dict of coordinate names and tensor sizes
    vars_in : tuple of the sizes of the tensors output by the model
    mapping_function : A function that takes an input coordinate dictionary and tuple of tensors, and outputs a dictionary of output variable tensors
    vars_out : dict of output variable tensor sizes"""
    def __init__(self, coords_in, vars_in, mapping_function):
        self.coords_in = coords_in
        self.vars_in = vars_in
        self.mapping_function = mapping_function
        test_coord_input = {}
        for name, size in coords_in.items():
            test_coord_input[name] = (torch.zeros(size))[None, ...]
        test_var_input = tuple((torch.zeros(size))[None, ...] for size in vars_in)
        test_output = mapping_function(test_coord_input, test_var_input)
        self.vars_out = {}
        for name, output in test_output:
            self.vars_out[name] = output[0, ...].size()

    def __call__(coords, variables):
        return self.mapping_function(coords, variables)

class Solution(torch.nn.Module):
    """The base class of all equation solutions.
       Contains an input map from named coordinates to model inputs,
       a model of some sort, and an output map from the model output
       to a set of named variables. Implements forward() method and 
       subclasses torch.nn.Module, so the entire setup can be treated
       as a single model from the point of view of torch. Note that the
       input and output maps are not trainable, by design.
       Not meant to be instanced directly, subclasses implementing
       specific archetectures should be created and instanced, this
       class just makes it possible to easily attach input and output
       maps to these models, and informs the models of the number and
       size of input and output tensors so that archetectures can be
       defined in a relatively problem-independent way."""
    def __init__(self, input_map, output_map):
        """Constructor
        
        input_map: An InputRemap object connecting named coordinates to model inputs
        output_map: An OutputRemap object connecting model outputs to named output varaibles"""
        super(Solution, self).__init__()
        self.in_map = input_map
        self.out_map = output_map
        self.setup_model(self.in_map.coords_out, self.out_map.vars_in)

    # do I have to declare the 'setup_model' function yet? Or the model function?
    
    def forward(self, coords): 
        model_input = self.in_map(coords)
        model_output = self.model(model_input)
        return self.out_map(coords, model_output)

### Model Implementations ###

class ClassicPINN(Solution):
    def setup_model(self, model_inputs, model_outputs):
        input_size = sum(size[0] for size in model_inputs)
        output_sizes = [size[0] for size in model_outputs]
        output_size = sum(output_sizes)
        u = torch.nn.Sequential(
            nn.Linear(input_size, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, output_size))
        def nn_eval(coords):
            input_vect = torch.cat(coords, 1)
            return torch.split(u(input_vect), output_sizes, dim=1)
        self.model = nn_eval
         
        

# For now, hard-code the co-location loss. It should take an argument for co-location weights, since some training methods use these (it's basically another NN... should it be seperate?)

# For now, implement the standard PINN loss and PINN boundary loss functions

# Finally, try to get the whole sympy -> equation loss thing going!
       
