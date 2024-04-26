import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('input_view', 0, 'View No. informative view, ACM:0, DBLP:1')
flags.DEFINE_float('weight_decay', 0.0001, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('fea_decay', 0.5, 'feature decay.')
flags.DEFINE_float('weight_R', 0.001, 'Weight for R loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('attn_drop', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('ffd_drop', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 50, 'number of iterations.')
flags.DEFINE_float('kl_decay', 0.1, 'kl loss decay.')


# def get_settings(Traindataname, Traindataname_truefeatures, Traindataname_truelabels, Testdataname, Testdataname_truefeatures, Testdataname_truelabels, iterations, kl_iterations, label_number, tol, pathwaytype):
#     re = {'Traindataname': Traindataname, 'Traindataname_truefeatures': Traindataname_truefeatures, 'Traindataname_truelabels': Traindataname_truelabels, 'Testdataname': Testdataname, 'Testdataname_truefeatures': Testdataname_truefeatures, 'Testdataname_truelabels': Testdataname_truelabels, 'iterations': iterations, 'kl_iterations': kl_iterations, 'label_number': label_number, 'tol': tol, 'pathwaytype': pathwaytype}
#     return re
