from input_data import load_data
from preprocessing import preprocess_graph
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
def get_placeholder(adjs_in, numView):
    placeholders = {
        'features': tf.placeholder(tf.float32),
        'adjs': tf.placeholder(tf.float32),
        # 'adjs': tf.placeholder(tf.float32, shape=(None, None, None)),
        'adjs_orig': tf.placeholder(tf.float32),
        # 'adjs_orig': tf.placeholder(tf.float32, shape=(None, None, None)),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'attn_drop': tf.placeholder_with_default(0., shape=()),
        'ffd_drop': tf.placeholder_with_default(0., shape=()),
        'pos_weights': tf.placeholder(tf.float32),
        'fea_pos_weights': tf.placeholder(tf.float32),
        'p': tf.placeholder(tf.float32),
        'norm': tf.placeholder(tf.float32),
        'true_labels': tf.placeholder(tf.float32),
        'predict_labels': tf.placeholder(tf.float32),

    }
    return placeholders
def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adjs']: adj_normalized})
    feed_dict.update({placeholders['adjs_orig']: adj})
    return feed_dict
def update(model, opt, sess, adj_norm, adj_label, features, placeholders, pos_weights, fea_pos_weights, norm, true_labels):

    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)

    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['pos_weights']: pos_weights})
    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})
    feed_dict.update({placeholders['true_labels']: true_labels})
    prediction = sess.run(model.embeddings2, feed_dict=feed_dict)
    feed_dict.update({placeholders['predict_labels']: prediction})
    reconstruct_loss = 0
    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    prediction = sess.run(model.embeddings2, feed_dict=feed_dict)
    return prediction, reconstruct_loss

def update_test(model, opt, sess, adj_norm, adj_label, features, placeholders, pos_weights, fea_pos_weights, norm):

    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['pos_weights']: pos_weights})
    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})
    emb_ind = sess.run(model.embeddings, feed_dict=feed_dict)
    return emb_ind
def update_test2(model, opt, sess, adj_norm, adj_label, features, placeholders, pos_weights, fea_pos_weights, norm):

    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['pos_weights']: pos_weights})
    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})
    emb_ind = sess.run(model.embeddings2, feed_dict=feed_dict)
    return emb_ind



def format_data(Traindataname_truefeatures, Traindataname_truelabels, Testdataname_truefeatures, Testdataname_truelabels, pathwaytype):
    print("Traindataname_truefeatures：%s Traindataname_truelabels：%s Testdataname_truefeatures：%s Testdataname_truelabels：%s  pathwaytype: %s" % (Traindataname_truefeatures, Traindataname_truelabels, Testdataname_truefeatures, Testdataname_truelabels, pathwaytype))
    Train_rownetworks, Test_rownetworks, Train_numView, Test_numView, Traindataname_merged_harmony_data, Testdataname_merged_harmony_data, Train_one_hot_labels_values, Test_one_hot_labels_values, label_number = load_data(Traindataname_truefeatures, Traindataname_truelabels, Testdataname_truefeatures, Testdataname_truelabels, pathwaytype)

    Train_adjs_orig = []
    for v in range(Train_numView):
        Train_adj_orig = Train_rownetworks[v]
        Train_adj_orig = Train_adj_orig - sp.dia_matrix((Train_adj_orig.diagonal()[np.newaxis, :], [0]), shape=Train_adj_orig.shape)
        Train_adjs_orig.append(Train_adj_orig)
    Train_adjs_label = Train_rownetworks
    Train_adjs_orig = np.array(Train_adjs_orig)
    Train_adjs = Train_adjs_orig
    Train_adjs_norm = preprocess_graph(Train_adjs)
    Train_num_nodes = Train_adjs[0].shape[0]
    Traindataname_merged_harmony_data = Traindataname_merged_harmony_data
    num_Traindataname_merged_harmony_data = Traindataname_merged_harmony_data.shape[1]
    Train_fea_pos_weights = float(Traindataname_merged_harmony_data.shape[0] * Traindataname_merged_harmony_data.shape[1] - Traindataname_merged_harmony_data.sum()) / Traindataname_merged_harmony_data.sum()
    Train_pos_weights = []
    Train_norms = []
    for v in range(Train_numView):
        Train_pos_weight = float(Train_adjs[v].shape[0] * Train_adjs[v].shape[0] - Train_adjs[v].sum()) / Train_adjs[v].sum()
        Train_norm = Train_adjs[v].shape[0] * Train_adjs[v].shape[0] / float((Train_adjs[v].shape[0] * Train_adjs[v].shape[0] - Train_adjs[v].sum()) * 2)
        Train_pos_weights.append(Train_pos_weight)
        Train_norms.append(Train_norm)
    Train_one_hot_labels_values = Train_one_hot_labels_values
    Train_feas = {'adjs': Train_adjs_norm, 'adjs_label': Train_adjs_label, 'num_features': num_Traindataname_merged_harmony_data, 'num_nodes': Train_num_nodes, 'true_labels':Train_one_hot_labels_values, 'pos_weights': Train_pos_weights, 'norms': np.array(Train_norms), 'adjs_norm':Train_adjs_norm, 'features': Traindataname_merged_harmony_data, 'fea_pos_weights': Train_fea_pos_weights, 'numView': Train_numView}

    Test_adjs_orig = []
    for v in range(Test_numView):
        Test_adj_orig = Test_rownetworks[v]
        Test_adj_orig = Test_adj_orig - sp.dia_matrix((Test_adj_orig.diagonal()[np.newaxis, :], [0]), shape=Test_adj_orig.shape)
        Test_adjs_orig.append(Test_adj_orig)
    Test_adjs_label = Test_rownetworks
    Test_adjs_orig = np.array(Test_adjs_orig)
    Test_adjs = Test_adjs_orig
    Test_adjs_norm = preprocess_graph(Test_adjs)
    Test_num_nodes = Test_adjs[0].shape[0]
    Testdataname_merged_harmony_data = Testdataname_merged_harmony_data
    num_Testdataname_merged_harmony_data = Testdataname_merged_harmony_data.shape[1]
    Test_fea_pos_weights = float(Testdataname_merged_harmony_data.shape[0] * Testdataname_merged_harmony_data.shape[1] - Testdataname_merged_harmony_data.sum()) / Testdataname_merged_harmony_data.sum()
    Test_pos_weights = []
    Test_norms = []
    for v in range(Test_numView):
        Test_pos_weight = float(Test_adjs[v].shape[0] * Test_adjs[v].shape[0] - Test_adjs[v].sum()) / Test_adjs[v].sum()
        Test_norm = Test_adjs[v].shape[0] * Test_adjs[v].shape[0] / float((Test_adjs[v].shape[0] * Test_adjs[v].shape[0] - Test_adjs[v].sum()) * 2)
        Test_pos_weights.append(Test_pos_weight)
        Test_norms.append(Test_norm)
    Test_one_hot_labels_values = Test_one_hot_labels_values
    Test_feas = {'adjs': Test_adjs_norm, 'adjs_label': Test_adjs_label, 'num_features': num_Testdataname_merged_harmony_data, 'num_nodes': Test_num_nodes, 'true_labels':Test_one_hot_labels_values, 'pos_weights': Test_pos_weights, 'norms': np.array(Test_norms), 'adjs_norm':Test_adjs_norm, 'features': Testdataname_merged_harmony_data, 'fea_pos_weights': Test_fea_pos_weights, 'numView': Test_numView}
    return Train_feas, Test_feas, label_number

def compute_q(model, opt, sess, adj_norm, adj_label, features, placeholders, pos_weights, fea_pos_weights, norm):

    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['pos_weights']: pos_weights})
    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})


    '''
    for key in feed_dict.keys():
        print('key', key)
        print('value', feed_dict[key])
    '''



    q = sess.run(model.cluster_layer_q, feed_dict=feed_dict)

    return q
def update_kl(model, opt, sess, adj_norm, adj_label, features, p, placeholders, pos_weights, fea_pos_weights, norm, idx,true_labels):
    # construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['pos_weights']: pos_weights})
    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})
    feed_dict.update({placeholders['true_labels']: true_labels})
    feed_dict.update({placeholders['p']: p})


    #feed_dict.update({placeholders['dropout']: 0})
    '''
    for key in feed_dict.keys():
        print('key', key)
        print('value', feed_dict[key])
    '''

    #feed_dict.update({placeholders['real_distribution']: z_real_dist})
    for j in range(5):
        _, kl_loss = sess.run([opt.opt_op_kl, opt.cost_kl], feed_dict=feed_dict)
    '''
    vars_embed = sess.run(opt.grads_vars, feed_dict=feed_dict)
    norms = []
    for n in range(vars_embed[0][0].shape[0]):
        norms.append(np.linalg.norm(vars_embed[0][0][n]))
    cluster_layer_q = sess.run(model.cluster_layer_q, feed_dict=feed_dict)
    y_pred = cluster_layer_q.argmax(1)
    idx_list = []
    for n in range(len(y_pred)):
        if y_pred[n]==idx:
            idx_list.append(n)
    norms = np.array(norms)
    norms_tmp = norms[idx_list]
    label = np.array(label)[idx_list]
    tmp_q = cluster_layer_q[idx_list][:, idx]
    print('idx', idx)
    fw = open('./norm_q.txt', 'w')
    for n in range(len(norms_tmp)):
        str1 = str(norms_tmp[n]) + ' ' + str(tmp_q[n]) + ' ' + str(label[n])
        fw.write(str1)
        fw.write('\n')
    fw.close()
    '''
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    avg_cost = kl_loss

    return emb,avg_cost