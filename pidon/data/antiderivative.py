import jax.numpy as jnp
from jax import random, vmap, jit
from jax.experimental.ode import odeint
from jax import config

# Define RBF kernel
def RBF(x1, x2, params):
    output_scale, length_scale = params
    diffs = jnp.expand_dims(x1 / length_scale, 1) - \
            jnp.expand_dims(x2 / length_scale, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)

# Geneate training data corresponding to one input sample
def generate_one_training_data(key, length_scale=0.2, m=100, P=1, Q=100):
    # Sample GP prior at a fine grid
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(key, (N,)))

    # Create a callable interpolation function  
    u_fn = lambda x, t: jnp.interp(t, X.flatten(), gp_sample)

    # Input sensor locations and measurements
    x = jnp.linspace(0, 1, m)
    u = vmap(u_fn, in_axes=(None,0))(0.0, x)

    # Output sensor locations and measurements
    y_train = random.uniform(key, (P,)).sort() 
    s_train = odeint(u_fn, 0.0, jnp.hstack((0.0, y_train)))[1:] # JAX has a bug and always returns s(0), so add a dummy entry to y and return s[1:]

    # Tile inputs
    u_train = jnp.tile(u, (P, 1))

    # training data for the residual
    u_r_train = jnp.tile(u, (Q, 1))
    y_r_train = jnp.linspace(0, 1, Q)
    s_r_train = vmap(u_fn, in_axes=(None,0))(0.0, y_r_train)

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train

# Geneate test data corresponding to one input sample
def generate_one_test_data(key, length_scale=0.2, m=100, P=100):
    # Sample GP prior at a fine grid
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(key, (N,)))

    # Create a callable interpolation function  
    u_fn = lambda x, t: jnp.interp(t, X.flatten(), gp_sample)

    # Input sensor locations and measurements
    x = jnp.linspace(0, 1, m)
    u = vmap(u_fn, in_axes=(None,0))(0.0, x)

    # Output sensor locations and measurements
    y = jnp.linspace(0, 1, P)
    s = odeint(u_fn, 0.0, y)

    # Tile inputs
    u = jnp.tile(u, (P, 1))

    return u, y, s 

# Geneate training data corresponding to N input sample
def generate_training_data(key, length_scale, N, m, P, Q):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    gen_fn = jit(lambda key: generate_one_training_data(key, length_scale, m, P, Q))
    u_train, y_train, s_train, u_r_train, y_r_train, s_r_train = vmap(gen_fn)(keys)

    u_train = jnp.float32(u_train.reshape(N * P,-1))
    y_train = jnp.float32(y_train.reshape(N * P,-1))
    s_train = jnp.float32(s_train.reshape(N * P,-1))

    u_r_train = jnp.float32(u_r_train.reshape(N * Q,-1))
    y_r_train = jnp.float32(y_r_train.reshape(N * Q,-1))
    s_r_train = jnp.float32(s_r_train.reshape(N * Q,-1))

    config.update("jax_enable_x64", False)
    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train

# Geneate test data corresponding to N input sample
def generate_test_data(key, length_scale, N, m, P):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    gen_fn = jit(lambda key: generate_one_test_data(key, length_scale, m, P))
    u, y, s = vmap(gen_fn)(keys)
    u = jnp.float32(u.reshape(N * P,-1))
    y = jnp.float32(y.reshape(N * P,-1))
    s = jnp.float32(s.reshape(N * P,-1))

    config.update("jax_enable_x64", False)
    return u, y, s
