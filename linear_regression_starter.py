import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd



data_file = 'assets/fire_theft.csv'

# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file
data = pd.read_csv(data_file)


# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
# Both have the type float32

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')


# Step 3: create weight and bias, initialized to 0
# name your variables w and b
w = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='bias')


# Step 4: predict Y (number of theft) from the number of fire
# name your variable Y_predicted
Y_predicted = X * w + b

# Step 5: use the square error as the loss function
# name your variable loss

loss = tf.square(Y - Y_predicted, name='loss')


# Step 6: using gradient descent with learning rate of 0.01 to minimize loss

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss)

# Phase 2: Train our model
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    # TO - DO
    sess.run(tf.global_variables_initializer())

    #initialize writer
    writer = tf.summary.FileWriter('./graphs/lin_reg', sess.graph)


    # Step 8: train the model
    for i in range(101):  # run 100 epochs
        total_loss = 0
        for x, y in data.values:
            # Session runs optimizer to minimize loss and fetch the value of loss. Name the received value as l
            # TO DO: write sess.run()
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})

            total_loss += l
        print("Epoch {0}: {1}".format(i, total_loss / len(data)))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w, b = sess.run([w, b])

# plot the results
X, Y = data['X'], data['Y']
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()