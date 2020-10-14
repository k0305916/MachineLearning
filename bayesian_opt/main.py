from bayes_opt import BayesianOptimization
import numpy as np

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    # x = int(x)
    # y = int(y)

    return -x ** 2 - (y - 1) ** 2 + 1

def black_box_function1(**kwargs):
    x = np.fromiter(kwargs.values(), dtype=float)
    return -x[0] ** 2 - (x[1] - 1) ** 2 + 1


# Bounded region of parameter space
# 好像是不能取到(-∞, +∞)
pbounds = {'x': (2, 4), 'y': (-3, 3)}

# optimizer = BayesianOptimization(
#     f=black_box_function,
#     pbounds=pbounds,
#     verbose=2,
#     random_state=1,
# )

# optimizer.maximize(
#     # init_points=2,
#     # n_iter=3,
# )

# print(optimizer.max)

optimizer = BayesianOptimization(
    f=black_box_function1,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

optimizer.maximize(
    # init_points=2,
    # n_iter=3,
)

print(optimizer.max)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

optimizer.set_bounds(new_bounds={"x": (-2, 3)})

optimizer.maximize(
    init_points=0,
    n_iter=5,
)