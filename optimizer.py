import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, model, preds_fuze, p, labels,  numView, pos_weights, fea_pos_weights, norm,true_labels, predict_labels):
        labels_sub = labels
        self.cost = 0
        self.cost_list = []
        all_variables = tf.trainable_variables()
        self.l2_loss = 0
        for var in all_variables:

            self.l2_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for v in range(numView-1):
            self.cost += 0.1*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                logits=tf.reshape(preds_fuze[v], [-1]), targets=tf.reshape(labels_sub[v], [-1]), pos_weight=pos_weights[v]))
        self.cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels, logits=predict_labels))
        self.cost += self.l2_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                           beta1=0.9, name='adam')  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        q = model.cluster_layer_q
        kl_loss = tf.reduce_sum(p * tf.log(p/q))
        self.cost_kl = self.cost + FLAGS.kl_decay* kl_loss
        self.opt_op_kl = self.optimizer.minimize(self.cost_kl)
               
    


