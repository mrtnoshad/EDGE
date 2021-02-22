# Morteza Noshad: EDGE on CNN for MNIST dataset
# Network design by Krzysztof Sopyła
#

# Network architecture:
# Five layer neural network, input layer 28*28= 784, output 10 (10 digits)
# Output labels uses one-hot encoding

# input layer             - X[batch, 784]
# 1 layer                 - W1[784, 200] + b1[200]
#                           Y1[batch, 200] 
# 2 layer                 - W2[200, 100] + b2[100]
#                           Y2[batch, 100] 
# 3 layer                 - W3[100, 60]  + b3[60]
#                           Y3[batch, 60] 
# 4 layer                 - W4[60, 30]   + b4[30]
#                           Y4[batch, 30] 
# 5 layer                 - W5[30, 10]   + b5[10]
# One-hot encoded labels    Y5[batch, 10]

# model
# Y = softmax(X*W+b)
# Matrix mul: X*W - [batch,784]x[784,10] -> [batch,10]

# Training consists of finding good W elements. This will be handled automaticaly by 
# Tensorflow optimizer

#import visualizations as vis
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
from random import randint, seed
from sklearn.neighbors import NearestNeighbors


NUM_ITERS=12000
DISPLAY_STEP=100
BATCH=100


# Download images and labels 
mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)

# mnist.test (10K images+labels) -> mnist.test.images, mnist.test.labels
# mnist.train (60K images+labels) -> mnist.train.images, mnist.test.labels

# Placeholder for input images, each data sample is 28x28 grayscale images
# All the data will be stored in X - tensor, 4 dimensional matrix
# The first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)


## Model


# layers sizes
C1 = 4  # first convolutional layer output depth
C2 = 8  # second convolutional layer output depth
C3 = 16 # third convolutional layer output depth

FC4 = 256  # fully connected layer

n_hidden_layers = 4

# weights - initialized with random values from normal distribution mean=0, stddev=0.1
# output of one layer is input for the next
def build_model(i):
    
    global Y1, Y2, Y3, Y4, Y, Ylogits, cross_entropy, correct_prediction, accuracy, train_step 

    tf.set_random_seed(i)

    # weights - initialized with random values from normal distribution mean=0, stddev=0.1

    # 5x5 conv. window, 1 input channel (gray images), C1 - outputs
    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, C1], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([C1], stddev=0.1))
    # 3x3 conv. window, C1 input channels(output from previous conv. layer ), C2 - outputs
    W2 = tf.Variable(tf.truncated_normal([3, 3, C1, C2], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([C2], stddev=0.1))
    # 3x3 conv. window, C2 input channels(output from previous conv. layer ), C3 - outputs
    W3 = tf.Variable(tf.truncated_normal([3, 3, C2, C3], stddev=0.1))
    b3 = tf.Variable(tf.truncated_normal([C3], stddev=0.1))
    # fully connected layer, we have to reshpe previous output to one dim, 
    # we have two max pool operation in our network design, so our initial size 28x28 will be reduced 2*2=4
    # each max poll will reduce size by factor of 2
    W4 = tf.Variable(tf.truncated_normal([7*7*C3, FC4], stddev=0.1))
    b4 = tf.Variable(tf.truncated_normal([FC4], stddev=0.1))

    # output softmax layer (10 digits)
    W5 = tf.Variable(tf.truncated_normal([FC4, 10], stddev=0.1))
    b5 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

    # flatten the images, unroll each image row by row, create vector[784] 
    # -1 in the shape definition means compute automatically the size of this dimension
    XX = tf.reshape(X, [-1, 784])


    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + b1, name='hidden1')

    k = 2 # max pool filter size and stride, will reduce input by factor of 2
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + b2)
    Y2 = tf.nn.max_pool(Y2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name='hidden2')

    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + b3)
    Y3 = tf.nn.max_pool(Y3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name='hidden3')

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * C3])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4, name='hidden4')

    #Y4 = tf.nn.dropout(Y4, pkeep)
    Ylogits = tf.matmul(Y4, W5) + b5
    Y = tf.nn.softmax(Ylogits)


    # we can also use tensorflow function for softmax
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100

                                                              
    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training, 
    learning_rate = 0.003
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


def get_hidden_layers(names):
    hidden_layers = []
    for name in names:
        #print('name: ',name)
        hidden_layers.append(tf.get_default_graph().get_tensor_by_name("%s:0" % name))
    return hidden_layers

# get mutual information for all hidden layers
def get_MI_EDGE(hiddens, ep_idx):
    mi_xt_list = []; mi_ty_list = []
    #hidden = hiddens[1]
    hidden_idx = 0
    for hidden in hiddens:
	    mi_xt, mi_ty = calc_MI_EDGE(hidden,hidden_idx ,ep_idx)
	    mi_xt_list.append(mi_xt)
	    mi_ty_list.append(mi_ty)
	    hidden_idx +=1

    return mi_xt_list, mi_ty_list

print(mnist.test.images.shape)
print(mnist.test.labels.shape)

T = 10000

X_MI= mnist.train.images[:T,:,:,:]
Y_MI= mnist.train.labels[:T,:]


## Mutual information computation

from EDGE_4_2_0 import EDGE
global dist0
dist0 = np.zeros(4)

def calc_MI_EDGE(hidden,layer_idx, ep_idx):
    global rho_0
   
    hidden = np.array(hidden)[:T,:]
    #print('calc_MI_EDGE',hidden.shape)
    N, d=hidden.shape[0],hidden.shape[1]
    #print(hidden.shape)
    X_reshaped = np.reshape(X_MI,[-1,784]) # vectorize X
    Y_reshaped = np.argmax(Y_MI, axis=1)# convert 10-dim data to class integer in [0,9]
    
    H = np.array(hidden)
    #d_temp = H.shape[1]*H.shape[2]*H.shape[3]
    hidden_reshaped = np.reshape(H,(N,-1))
    print('d_temp: ', hidden_reshaped.shape[1])

    # Normalize hidden
    smoothness_vector_xt = np.array([0.8, 1.0, 1.2, 1.4])
    smoothness_vector_ty = np.array([0.2, 0.3, 0.34, 0.52])

    mi_xt_py = EDGE(X_reshaped, hidden_reshaped,U=20, L_ensemble=10, gamma=[0.2,  smoothness_vector_xt[layer_idx]], epsilon_vector= 'range') #,U=20, gamma=[0.2,  2*smoothness_vector[layer_idx]], epsilon=[0.2,r*0.2], hashing='p-stable') 
    mi_ty_py = EDGE(Y_reshaped, hidden_reshaped,U=10, L_ensemble=10, gamma=[0.0001, smoothness_vector_ty[layer_idx]],epsilon=[0.2, 0.2], epsilon_vector= 'range')

    #mi_xt_py = EDGE(X_reshaped, hidden,U=20, gamma=[0.2,  2*smoothness_vector[layer_idx]], epsilon=[0.2,r*0.2], hashing='p-stable') 
    #mi_ty_py = EDGE(Y_reshaped, hidden,U=10, gamma=[0.0001, smoothness_vector[layer_idx]], epsilon=[0.2, 0.2], hashing='p-stable')

    return mi_xt_py, mi_ty_py



####### Run with computation of MI ######
def train_with_mi(random_idx):
    print('train_with_mi')


    build_model(random_idx)


    # Initializing the variables
    
    mi_xt_all = []; mi_ty_all = []; epochs = []
    hidden_layer_names = ['hidden%s' % i for i in range(1,n_hidden_layers+1)]
    print(hidden_layer_names)

    train_losses = list()
    train_acc = list()
    test_losses = list()
    test_acc = list()

    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        #print('session')
        #sess.run(init)
        sess.run(tf.global_variables_initializer()) # initialization
        
        #print('beFor')
        for i in range(NUM_ITERS+1):
            #print('epoch: ', i )
            # training on batches of 100 images with 100 labels
            batch_X, batch_Y = mnist.train.next_batch(BATCH)

            # Print summary
            if i%DISPLAY_STEP == 0:
                # compute training values for visualisation
                acc_trn, loss_trn = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
                acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})

                print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))

                train_losses.append(loss_trn)
                train_acc.append(acc_trn)
                test_losses.append(loss_tst)
                test_acc.append(acc_tst)

            # the backpropagationn training step
            sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, pkeep: 0.75})

            # Compute MI
            #
            q1 = 1
            q2 = 1
            A_ = i <= 10 and i % q1 == 0
            A0 = i > 10 and i <= 100 and i % (3*q2) == 0     
            A1 = i > 100 and i <= 1000 and i % (25*q2) == 0    
            A2 = i > 1000 and i <= 2000 and i % (50*q2) == 0
            A3 = i > 2000 and i <= 4000 and i % (200*q2) == 0
            A4 = i > 4000 and i % (400*q2) == 0

            #if A0 or A1 or A2:
            if   A_ or A0 or A1 or A2 or A3 or A4:
                
                _, hidden_layers = sess.run([train_step,
                                             get_hidden_layers(hidden_layer_names)],
                                             feed_dict={X: X_MI, Y_: Y_MI, pkeep: 1.0})
                #print(len(hidden_layers), len(hidden_layers[0]), len(hidden_layers[0][0]))
                
                #H = np.array(hidden_layers[0])
                #print('hidden_layers', H.shape)
                mi_xt, mi_ty = get_MI_EDGE(hidden_layers, i)
                
                print('MI(X;T): ',mi_xt,'MI(Y;T): ', mi_ty)
                
                mi_xt_all.append(mi_xt)
                mi_ty_all.append(mi_ty)
                #epochs.append(epoch)
                
    return np.array(mi_xt_all), np.array(mi_ty_all)

#title = "MNIST 2.1 5 layers relu adam"
#vis.losses_accuracies_plots(train_losses,train_acc,test_losses, test_acc,title,DISPLAY_STEP)



import multiprocessing
from multiprocessing import Pool

num_cores = multiprocessing.cpu_count()
Rep = 23
inputs = range(Rep)

with Pool(num_cores) as p:
    #mi_xt_all, mi_ty_all = p.map(gen_MI_all_itirations, inputs)
    mi_all = p.map(train_with_mi, inputs)
#mi_xt_all, mi_ty_all= Parallel(n_jobs=num_cores)(delayed(gen_MI_all_itirations)(i) for i in inputs)

np.save('mi_all', mi_all)

