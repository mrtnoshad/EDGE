
# Network architecture:
# Five layer neural network, input layer 28*28= 784, output 10 (10 digits)
# Output labels uses one-hot encoding

# Training consists of finding good W elements. This will be handled automaticaly by 
# Tensorflow optimizer

#import visualizations as vis
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
from random import randint, seed
from sklearn.neighbors import NearestNeighbors


NUM_ITERS=20000
DISPLAY_STEP=100
BATCH=128


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


## Model


# layers sizes
L1 = 1024
L2 = 20
L3 = 20
L4 = 20
L5 = 10
n_hidden_layers = 4

# weights - initialized with random values from normal distribution mean=0, stddev=0.1
# output of one layer is input for the next
def build_model(i):
    
    global Y1, Y2, Y3, Y4, Y, Ylogits, cross_entropy, correct_prediction, accuracy, train_step 

    tf.set_random_seed(i)

    W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([L1]))

    W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
    b2 = tf.Variable(tf.zeros([L2]))

    W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([L3]))

    W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
    b4 = tf.Variable(tf.zeros([L4]))

    W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1))
    b5 = tf.Variable(tf.zeros([L5]))


    # flatten the images, unrole eacha image row by row, create vector[784] 
    # -1 in the shape definition means compute automatically the size of this dimension
    XX = tf.reshape(X, [-1, 784])

    # Define model
    Y1 = tf.nn.tanh(tf.matmul(XX, W1) + b1, 'hidden1')
    Y2 = tf.nn.tanh(tf.matmul(Y1, W2) + b2, 'hidden2')
    Y3 = tf.nn.tanh(tf.matmul(Y2, W3) + b3, 'hidden3')
    Y4 = tf.nn.tanh(tf.matmul(Y3, W4) + b4, 'hidden4')
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
	    H = np.array(hidden)
	        #print('get_MI_EDGE ',H.shape)
	    mi_xt, mi_ty = calc_MI_EDGE(hidden,hidden_idx,ep_idx)
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

def calc_MI_EDGE(hidden, layer_idx ,ep_idx):
    global rho_0
   
    hidden = np.array(hidden)[:T,:]
    #print('calc_MI_EDGE',hidden.shape)
    d=hidden.shape[1]
    #print(hidden.shape)
    X_reshaped = np.reshape(X_MI,[-1,784]) # vectorize X
    Y_reshaped = np.argmax(Y_MI, axis=1)# convert 10-dim data to class integer in [0,9]
 
    dist=0
    if ep_idx <=20:
        dist0[layer_idx] = av_distance(hidden)
        r = 1
    else:
        dist = av_distance(hidden)
        r = dist / dist0[layer_idx]

    print('ep_idx and hidden dim and r', ep_idx, hidden.shape[1] ,r, dist)    

     
    # Normalize hidden
    #hidden = hidden/r
    smoothness_vector_xt = np.array([0.8, 1.0, 1.2, 1.8])
    smoothness_vector_ty = np.array([0.4, 0.5, 0.6, 0.8])

    mi_xt_py = EDGE(X_reshaped, hidden,U=20, L_ensemble=10, gamma=[0.2,  smoothness_vector_xt[layer_idx]], epsilon_vector= 'range') #,U=20, gamma=[0.2,  2*smoothness_vector[layer_idx]], epsilon=[0.2,r*0.2], hashing='p-stable') 
    mi_ty_py = EDGE(Y_reshaped, hidden,U=10, L_ensemble=10, gamma=[0.0001, smoothness_vector_ty[layer_idx]], epsilon=[0.2,0.2], epsilon_vector= 'range')

    return mi_xt_py, mi_ty_py

# Find average distances between points
from numpy import linalg as LA
def av_distance(X):

    r = 1000

    N = X.shape[0]

    np.random.seed(1234)
    T1= np.random.choice(range(N), size=2*r)[:r]
    T2= np.random.choice(range(N), size=2*r)[r:]
    np.random.seed()
    D = LA.norm(X[T2,:] - X[T1,:], ord=2, axis=1)
    d = np.mean(D)

    return d


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
                acc_trn, loss_trn = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
                acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})

                print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))

                train_losses.append(loss_trn)
                train_acc.append(acc_trn)
                test_losses.append(loss_tst)
                test_acc.append(acc_tst)

            # the backpropagationn training step
            sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

            # Compute MI
            #
            q = 1
            A_ = i <= 10 and i % 1 == 0
            A0 = i > 10 and i <= 100 and i % (3*q) == 0     
            A1 = i > 100 and i <= 1000 and i % (25*q) == 0    
            A2 = i > 1000 and i <= 2000 and i % (50*q) == 0
            A3 = i > 2000 and i <= 4000 and i % (200*q) == 0
            A4 = i > 4000 and i % (400*q) == 0

            #if A0 or A1 or A2:
            if   A_ or A0 or A1 or A2 or A3 or A4:
                
                _, hidden_layers = sess.run([train_step,
                                             get_hidden_layers(hidden_layer_names)],
                                             feed_dict={X: X_MI, Y_: Y_MI})
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

np.save('mi_all', mi_all)

