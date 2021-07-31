#!/usr/bin/env python
"""
Testing out the PINN method on Burger's equation.
(Just a re-creation of work from the original PINN paper).
"""

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from pyDOE import lhs
import time

seed = 84
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Lets be crazy and not do this with classes or any fanciness.
# Time for some bare-bones code to better elucidate the core ideas!

# Mysterious ML sage wisdom,
# Apparently this is a "good" initialization for symmetric saturating activation functions:
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

# First, construct & initialize the network:
layers = [2, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 32, 32, 32, 16, 1] # Layer sizes
#layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1] # Suposedly they did well with this? Doesn't work well for me.
weights = []
biases = []
for l in range(0, len(layers)-1):
    W = xavier_init(size=[layers[l], layers[l+1]])
    b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32) # ???
    weights.append(W)
    biases.append(b)

# Now, define function for a forward pass
# (Isn't there some tf short-hand pre-implemented function for this sort of thing?)
def forward_pass(X, weights, biases):
    H = X
    for l in range(0, len(weights)-1):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b) # No activation on final layer I guess
    return Y

# Define domain
x_min = -1.0
x_max = +1.0
t_min = 0.
t_max = 1.

# Define solution evaluation function, and equation satisfaction constraint
def u(t, x):
    u = forward_pass(tf.stack([t, x], axis=1), weights, biases)
    return tf.squeeze(u)

def u_x(t, x):
    u_val = u(t, x)
    u_x = tf.gradients(u_val, x)[0]
    return u_x

def u_t(t, x):
    u_val = u(t, x)
    u_t = tf.gradients(u_val, t)[0]
    return u_t

def f(t, x):
    u_val = u(t, x)
    u_t = tf.gradients(u_val, t)[0]
    u_x = tf.gradients(u_val, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    f_val = u_t + u_val*u_x - (0.01/np.pi)*u_xx # Viscosity coefficient hard-coded here
    return f_val

# Create place-holders for input & temporary variables
with tf.name_scope("Placeholders"):
    t_int = tf.compat.v1.placeholder(tf.float32, shape=[None])
    x_int = tf.compat.v1.placeholder(tf.float32, shape=[None])
    t_init = tf.compat.v1.placeholder(tf.float32, shape=[None])
    x_init = tf.compat.v1.placeholder(tf.float32, shape=[None])
    u_init = tf.compat.v1.placeholder(tf.float32, shape=[None])
    t_boundary_xmin = tf.compat.v1.placeholder(tf.float32, shape=[None])
    x_boundary_xmin = tf.compat.v1.placeholder(tf.float32, shape=[None])
    t_boundary_xmax = tf.compat.v1.placeholder(tf.float32, shape=[None])
    x_boundary_xmax = tf.compat.v1.placeholder(tf.float32, shape=[None])


# Create tf graphs for calculating stuff
u_init_pred = u(t_init, x_init)
u_boundary_xmin_pred = u(t_boundary_xmin, x_boundary_xmin)
u_boundary_xmax_pred = u(t_boundary_xmax, x_boundary_xmax)
f_pred = f(t_int, x_int)


# Define initial conditions
def u_0(x):
    return -np.sin(np.pi*((2.*(x-x_min)/(x_max-x_min))-1.))

# Define loss function and optimization algorithm.
#To-do

initial_loss = tf.reduce_mean(tf.square(u_init_pred - u_init))
interior_loss = tf.reduce_mean(tf.square(f_pred))
boundary_loss = tf.reduce_mean(tf.square(u_boundary_xmin_pred) + tf.square(u_boundary_xmax_pred))

loss1 = 100*initial_loss + 10*boundary_loss
loss2 = 100*initial_loss + interior_loss + 10*boundary_loss

# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,
#                                                    method='L-BFGS-B',
#                                                    options={'maxiter':40000,
#                                                             'maxfun':100000,
#                                                             'maxcor': 50,
#                                                             'maxls': 50,
#                                                             'ftol': 1.0*np.finfo(float).eps})
optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
opt1 = optimizer.minimize(loss1, var_list=tf.trainable_variables())
opt2 = optimizer.minimize(loss2, var_list=tf.trainable_variables())

# Generate training / regularization data
#To-do
N_t = 100
N_x = 100
N_consistency = 10000
t = np.linspace(t_min, t_max, N_t)
x = np.linspace(x_min, x_max, N_x)

lb = np.array([t_min, x_min])
ub = np.array([t_max, x_max])

# Train

# stack of t, x for each sample
X_initial_train = np.stack((np.zeros(N_x), # t for each sample
                            np.linspace(x_min, x_max, N_x))) # x for each sample
Y_initial_train = u_0(X_initial_train[1,:])

X_interior_train = lb + (ub - lb)*lhs(2, N_consistency)

num_batches = 1
X_interior_train_batches = np.array_split(X_interior_train, num_batches)

tf_dict1 = {t_init: X_initial_train[0,:],
            x_init: X_initial_train[1,:],
            u_init: Y_initial_train,
            t_boundary_xmin: t,
            x_boundary_xmin: x_min*np.ones(N_t),
            t_boundary_xmax: t,
            x_boundary_xmax: x_max*np.ones(N_t)}

tf_dict2 = []
for i in range(num_batches):
	tf_dict2.append({t_init: X_initial_train[0,:],
                     x_init: X_initial_train[1,:],
                     u_init: Y_initial_train,
                     t_int: X_interior_train_batches[i][:, 0],
                     x_int: X_interior_train_batches[i][:, 1],
                     t_boundary_xmin: t,
                     x_boundary_xmin: x_min*np.ones(N_t),
                     t_boundary_xmax: t,
                     x_boundary_xmax: x_max*np.ones(N_t)})

# Initialize variables, create session:
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

start_time = time.time()

print("Pretraining:")
for i in range(1500):
	_, loss_num = sess.run([opt1, loss1], feed_dict=tf_dict1)
#	if i % 10 == 0:
#		print("Epoch:{} Loss: {}".format(i, loss_num))

#u_init_pred_result = sess.run(u_init_pred, {x_init:x, t_init:np.zeros(N_x)})
#u_pred_result_1 = sess.run(u_init_pred, {x_init:x, t_init:0.1*np.ones(N_x)})
#u_pred_result_2 = sess.run(u_init_pred, {x_init:x, t_init:0.2*np.ones(N_x)})
#u_pred_result_3 = sess.run(u_init_pred, {x_init:x, t_init:0.3*np.ones(N_x)})
#u_pred_result_4 = sess.run(u_init_pred, {x_init:x, t_init:0.4*np.ones(N_x)})
#u_pred_result_5 = sess.run(u_init_pred, {x_init:x, t_init:0.5*np.ones(N_x)})

#plt.plot(x, u_init_pred_result, label="NN")
#plt.plot(x, u_0(x), label="Initial Condition")
#plt.plot(x, u_pred_result_1, label="NN at t=0.1")
#plt.plot(x, u_pred_result_2, label="NN at t=0.2")
#plt.plot(x, u_pred_result_3, label="NN at t=0.3")
#plt.plot(x, u_pred_result_4, label="NN at t=0.4")
#plt.plot(x, u_pred_result_5, label="NN at t=0.5")
#plt.legend()
#plt.show()

print("Training:")
for i in range(0, 100//num_batches):
    loss_num = 0.
    # Iterate over mini-batches
    for j in range(num_batches):
        _, temp_loss_num = sess.run([opt2, loss2], feed_dict=tf_dict2[j])
        loss_num += temp_loss_num
    #
#    if i % 10 == 0:
#        print("Epoch:{} Loss: {}".format(i, loss_num/num_batches))
#        if loss_num/num_batches < 0.01:
#        	break
    # optimizer.minimize(sess, feed_dict=tf_dict,
    #                    fetches=[loss, interior_loss, initial_loss, boundary_loss])

end_time = time.time()

print(end_time - start_time)
# Display

# Do a simple forward pass
u_init_pred_result = sess.run(u_init_pred, {x_init:x, t_init:np.zeros(N_x)})
u_pred_result_1 = sess.run(u_init_pred, {x_init:x, t_init:0.1*np.ones(N_x)})
u_pred_result_2 = sess.run(u_init_pred, {x_init:x, t_init:0.2*np.ones(N_x)})
u_pred_result_3 = sess.run(u_init_pred, {x_init:x, t_init:0.3*np.ones(N_x)})
u_pred_result_4 = sess.run(u_init_pred, {x_init:x, t_init:0.4*np.ones(N_x)})
u_pred_result_5 = sess.run(u_init_pred, {x_init:x, t_init:0.5*np.ones(N_x)})

plt.plot(x, u_init_pred_result, label="NN")
plt.plot(x, u_0(x), label="Initial Condition")
plt.plot(x, u_pred_result_1, label="NN at t=0.1")
plt.plot(x, u_pred_result_2, label="NN at t=0.2")
plt.plot(x, u_pred_result_3, label="NN at t=0.3")
plt.plot(x, u_pred_result_4, label="NN at t=0.4")
plt.plot(x, u_pred_result_5, label="NN at t=0.5")
plt.legend()
plt.show()

