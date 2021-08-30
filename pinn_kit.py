#!/usr/bin/env python
import torch
from torch import nn
#from torch.nn import functional as F #Not being used?
from pyDOE import lhs
import numpy as np
import sympy as sp

import typing
from typing import Dict, Tuple, Callable

from abc import ABC, abstractmethod

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

# Custom type aliases for type hinting
SymbolDefs = Dict[str, torch.Size]
SymbolValues = Dict[str, torch.tensor]

class InputRemap:
    """Defines a map between input coordinates and the model input.
    This is useful for inducing a natural topology that can allow automatic
    boundary condition solution.

    :ivar coords_in: A set of named coordinates and their tensor sizes.
    :type coords_in: dict of coordinate names, torch.Size objects
    :ivar mapping_function: A function that remaps input coordinates to a set of tensors.
    :type mapping_function: Function taking a dict of torch.tensor, returning a tuple of torch.tensor
    :ivar coords_out: A set of unnamed output tensor sizes
    :type coords_out: tuple of torch.Size"""
    def __init__(self, coords_in: SymbolDefs, mapping_function: Callable[[SymbolValues], Tuple[torch.tensor, ...]]) -> None:
        """Constructor.

        :param coords_in: A set of named coordinates and their tensor sizes.
        :type coords_in: dict of coordinate names, torch.Size objects
        :param mapping_function: A function that remaps input coordinates to a set of tensors.
        :type mapping_function: Function taking a dict of torch.tensor, returning a tuple of torch.tensor"""
        self.coords_in = coords_in
        self.mapping_function = mapping_function
        test_input = {}
        for name, size in coords_in.items():
            test_input[name] = (torch.zeros(size))[None, ...]
        test_output = mapping_function(test_input)
        self.coords_out = tuple(out_coord[0, ...].size() for out_coord in test_output)

    def __call__(self, coords: SymbolValues) -> Tuple[torch.tensor, ...]:
        """Apply the remapping to a set of symbols with tensor values, provided by a dictionary."""
        return self.mapping_function(coords)

class OutputRemap:
    """Defines a coordinate-dependent map between model output and the solution space.
    This is useful for imposing a solution ansatz that automatically satisfies boundary
    or junction conditions.

    :ivar coords_in: A set of named coordinates and their tensor sizes.
    :type coords_in: dict of coordinate names, torch.Size objects
    :ivar vars_in: The sizes of the tensors that the remap takes as input.
    :type vars_in: tuple of torch.Size
    :ivar mapping_function: A function that remaps input tensors to a collection of named tensor-valued output variables in a manner that may be dependent on the values of the input coordinates.
    :type mapping_function: Function taking two dicts of torch.tensor (coordinates and model output), returning a tuple of torch.tensor
    :ivar vars_out: A collection of names and sizes for the output tensors of the map.
    :type vars_out: dict of tensor.Size"""
    def __init__(self, coords_in: SymbolDefs, vars_in: Tuple[torch.Size, ...], mapping_function: Callable[[SymbolValues, Tuple[torch.tensor, ...]], SymbolValues]) -> None:
        """Constructor.

        :param coords_in: A set of named coordinates and their tensor sizes.
        :type coords_in: dict of coordinate names, torch.Size objects
        :param vars_in: The sizes of the tensors that the remap takes as input.
        :type vars_in: tuple of torch.Size
        :param mapping_function: A function that remaps input tensors to a collection of named tensor-valued output variables in a manner that may be dependent on the values of the input coordinates.
        :type mapping_function: Function taking two dicts of torch.tensor (coordinates and model output), returning a tuple of torch.tensor"""
        self.coords_in = coords_in
        self.vars_in = vars_in
        self.mapping_function = mapping_function
        test_coord_input = {}
        for name, size in coords_in.items():
            test_coord_input[name] = (torch.zeros(size))[None, ...]
        test_var_input = tuple((torch.zeros(size))[None, ...] for size in vars_in)
        test_output = mapping_function(test_coord_input, test_var_input)
        self.vars_out = {}
        for name, output in test_output.items():
            self.vars_out[name] = output[0, ...].size()

    def __call__(self, coords: SymbolValues, variables: Tuple[torch.tensor, ...]) -> SymbolValues:
        """Apply the remap given a dictionary of named tensor coordinates and a tuple of tensors (presumably the model output)."""
        return self.mapping_function(coords, variables)

class Solution(torch.nn.Module, ABC):
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
       defined in a relatively problem-independent way.

       :ivar in_map: A map from the named coordinates to the model input tensors.
       :type in_map: InputRemap
       :ivar out_map: A map from the model output tensors to the named output variable tensors.
       :type out_map: OutputRemap"""
    def __init__(self, input_map: InputRemap, output_map: OutputRemap) -> None:
        """Constructor.

       :param input_map: A map from the named coordinates to the model input tensors.
       :type input_map: InputRemap
       :param output_map: A map from the model output tensors to the named output variable tensors.
       :type output_map: OutputRemap"""
        super(Solution, self).__init__()
        self.in_map = input_map
        self.out_map = output_map
        self.setup_model(self.in_map.coords_out, self.out_map.vars_in)

    @abstractmethod
    def setup_model(self, model_inputs: Tuple[torch.Size, ...], model_outputs: Tuple[torch.Size, ...]) -> None:
        """The setup routine for the central trainable model of the solution.
        Given a specification of the order and sizes of the tensors output by the in_map and expected by the out_map, a model should be constructed that connects to these inputs and outputs.
        This routine defines the architecture that represents the solution, so it is left unimplemented on this abstract base class, and can be overloaded in any subclass to define general PINN architectures.

        :param model_inputs: The sizes of the collection of tensors that will be fed in to the model.
        :type model_inputs: tuple of torch.Size
        :param model_outputs: The sizes of the collection of tensors that the model is expected to produce.
        :type model_outputs: tuple of torch.Size"""
        raise NotImplementedError

    def forward(self, coords: SymbolValues) -> SymbolValues: 
        model_input = self.in_map(coords)
        model_output = self.model(model_input)
        return self.out_map(coords, model_output)


class Equation:
    """This class represents equations. Every equation has a symbolic form and a Solution to which it applies.
    Perhapse confusingly, in this case the 'solution' may not actually solve the equation, but it should after
    training if the solution was sufficiently trained using the provided f() function.
    The function f() takes in coordinate values, evaluates the solution on them to get variable values, and
    uses these tensors and their derivatives to evaluate the sum of all terms of the equation. When the equation
    is satisfied, all the components of this sum should be zero, but the exact manner by which this is enforced is
    left up to the training and loss functions that call f().

    :param solution: The solution function that should, after training, satisfy this equation.
    :type solution: A subclass of Solution
    :param symbolic_expression: The sum of all terms in the equation, which should equal zero when satisfied.
    :type symbolic_expression: A sympy Expr object"""
    def __init__(self, solution: Solution, equation: sp.Expr) -> None:
        """Constructor.

        :ivar solution: A trainable solution model. Its named inputs and outputs must include all variables used in this equation.
        :type solution: A subclass of Solution
        :ivar equation: A symbolic expression of the sum of all terms in the equation. Variable symbol names must match those used by the solution.
        :type equation: A sympy Expr object"""
        self.solution = solution
        self.symbolic_expression = equation

        # To-do: check that the variables in the equation are all accounted for in the solution
        # To-do: define a custom 'torch expression' type that has .func and .args methods just like sympy Expr
        # To-do: define a dictionary of torch functions indexed by the corresponding sympy type. The derivative ones in particular will need to be custom implementations using autograd
        # To-do: in the constructor, append to a copy of this dictionary entries that connect the names of the coordinates and variables to functions that return local tensor variables. This way, we only have to hash the coordinates once, and apply the model once, and those things can then be stored locally so we avoid re-hashing or re-evaluation of the model during the evaluation of the torch expression tree that will happen when f() is called.
        # To-do: Use this dictionary to walk the symbolic equation expression tree and build a corresponding torch expression tree, whith the custom classes.
        # When f is called, simply assign the coordinates to the right internal variables, evaluate the model and assign its outputs to the right internal variables, and execute the torch expression tree to produce an output.

# To-do: implement an 'Action' class in the vein of Equation that evaluates a lagrangian density. It can maybe implement some integration methods, certainly monte carlo, but also methods that assume structured samples and/or take advantage of access to derivatives or model evaluations.

# It it's not possible to train a model to output the action integral directly, because you need access to the internal solution values to ensure that the action obeyed the right formula. The best that could be done is to have a solution model, and a seperate NN that integrates, but training a net to integrate seems risky and unlikely to provide a speedup over standard methods, which would be necessary anyway to evaluate the training, so I see no benefit to such an architecture at present.
# However, what *could* be done is to define a model that produces integrated fluxes for specific equations where such a formulation is possible. This could be used in conjunction with input/output remap based schemes to guarantee global conservation of certain quantites, which may be useful to avoid drifting of bulk quantities during training. Unfortunately, in a continuous domain "local flux conservation" has no distinct meaning from exact solution of the equations everywhere, so this would be a weaker constraint than that of finite-volume flux conservative methods.

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

def ColocationLoss(f: Callable[[SymbolValues], torch.tensor], points: SymbolValues, weights: torch.tensor) -> float:
    return torch.mean(torch.square(weights*f(points)))


# For now, implement the standard PINN loss and PINN boundary loss functions

# Finally, try to get the whole sympy -> equation loss thing going!
