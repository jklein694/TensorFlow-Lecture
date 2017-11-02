# What’s a tensor? 
**An n-dimensional array** 
- 0-d tensor: scalar (number) 
- 1-d tensor: vector 
- 2-d tensor: matrix

import tensorflow as tf
a = tf.add(3, 5)
Why x, y?
TF automatically names the nodes when you don’t
explicitly name them.
x = 3
y = 5

### Optimizer
**It checks with the optimizer**

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

_, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y})

Session looks at all trainable variables that loss depends on and update them

Session looks at all trainable variables that optimizer depends on and update them

**More Optimizers**

tf.train.GradientDescentOptimizer

tf.train.AdagradOptimizer

tf.train.MomentumOptimizer

tf.train.AdamOptimizer

tf.train.ProximalGradientDescentOptimizer

tf.train.ProximalAdagradOptimizer

tf.train.RMSPropOptimizer

And more


### Trainable variables
tf.Variable(initial_value=None, **trainable=True**, collections=None,
validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None,
expected_shape=None, import_scope=None)