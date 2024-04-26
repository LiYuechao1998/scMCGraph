from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, ClusteringLayer
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# 配置 GPU 设备
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
    def _build(self):
        raise NotImplementedError
    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
    def fit(self):
        pass
    def predict(self):
        pass
class ARGA(Model):
    def __init__(self, placeholders, numView, num_features, label_number, **kwargs):
        super(ARGA, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.adjs = placeholders['adjs']
        self.dropout = placeholders['dropout']
        # self.features_nonzero = features_nonzero
        self.num_features = num_features
        self.label_number = label_number
        self.numView = numView
        self.build()
    def _build(self):
        with tf.variable_scope('Encoder', reuse=None):
            self.hidden1 = GraphConvolution(input_dim=self.num_features,
                                            output_dim=FLAGS.hidden1,
                                            adj=self.adjs[self.numView-1],
                                            # adj=self.adjs[0],
                                            act=lambda x: x,
                                            # act=tf.nn.softmax,
                                            dropout=self.dropout,
                                            logging=self.logging,
                                            name='e_dense_1_' + str(self.numView))(self.inputs)
            # self.noise = gaussian_noise_layer(self.hidden1, 0.1)
            self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                               output_dim=FLAGS.hidden2,
                                               adj=self.adjs[self.numView-1],
                                               # adj=self.adjs[0],
                                               act=lambda x: x,
                                               dropout=self.dropout,
                                               logging=self.logging,
                                               name='e_dense_2_' + str(self.numView))(self.hidden1)
            self.embeddings2 = tf.layers.dense(inputs=self.embeddings, units=self.label_number, activation=tf.nn.softmax,
                                               name='e_dense_3_' + str(self.numView))
        self.cluster_layer = ClusteringLayer(input_dim=FLAGS.hidden2, n_clusters=self.label_number, name='clustering')
        self.cluster_layer_q = self.cluster_layer(self.embeddings)
        self.reconstructions_fuze = []
        for v in range(self.numView-1):
            view_reconstruction = InnerProductDecoder(input_dim=FLAGS.hidden2, name='e_weight_multi_', v=v,
                                                      act=lambda x: x,
                                                      logging=self.logging)(
                self.embeddings)  # 原矩阵* 16*16的W矩阵  *原矩阵的转置去构建adjs_or 原始图信息
            self.reconstructions_fuze.append(view_reconstruction)

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise
