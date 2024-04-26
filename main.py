import settings
import tensorflow as tf
from constructor import format_data, get_placeholder, update, update_test, compute_q, update_kl, update_test2
from model import ARGA
from optimizer import OptimizerAE
import numpy as np
from metrics import predicting_metrics
from sklearn.cluster import KMeans
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
flags = tf.app.flags
FLAGS = flags.FLAGS
def count_num(labels):
    label_num = {}
    for label in labels:
        if label not in label_num:
            label_num[label] = 1
        else:
            label_num[label] += 1
    return label_num
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
def get_settings(Traindataname, Traindataname_truefeatures, Traindataname_truelabels, Testdataname, Testdataname_truefeatures, Testdataname_truelabels, iterations, kl_iterations, tol, pathwaytype):
    re = {'Traindataname': Traindataname, 'Traindataname_truefeatures': Traindataname_truefeatures, 'Traindataname_truelabels': Traindataname_truelabels, 'Testdataname': Testdataname, 'Testdataname_truefeatures': Testdataname_truefeatures, 'Testdataname_truelabels': Testdataname_truelabels, 'iterations': iterations, 'kl_iterations': kl_iterations, 'tol': tol, 'pathwaytype': pathwaytype}
    return re
loss = []
NMIs = []
class Clustering_Runner():
    def __init__(self, settings):
        print("Traindataname: %s, Traindataname_truefeatures: %s, Traindataname_truelabels: %s,Testdataname: %s, Testdataname_truefeatures: %s, Testdataname_truelabels: %s, number of iteration: %5d, number of kl_iterations: %5d, tol: %f, pathwaytype: %s" % (settings['Traindataname'], settings['Traindataname_truefeatures'], settings['Traindataname_truelabels'], settings['Testdataname'], settings['Testdataname_truefeatures'], settings['Testdataname_truelabels'], settings['iterations'], settings['kl_iterations'], settings['tol'], settings['pathwaytype']))
        self.Traindataname = settings['Traindataname']
        self.Traindataname_truefeatures = settings['Traindataname_truefeatures']
        self.Traindataname_truelabels = settings['Traindataname_truelabels']
        self.Testdataname = settings['Testdataname']
        self.Testdataname_truefeatures = settings['Testdataname_truefeatures']
        self.Testdataname_truelabels = settings['Testdataname_truelabels']
        self.iterations = settings['iterations']
        self.kl_iterations = settings['kl_iterations']
        # self.label_number = settings['label_number']
        self.tol = settings['tol']
        self.pathwaytype = settings['pathwaytype']
    def erun(self):
        tf.reset_default_graph()
        Trainfeas, Testfeas, label_number = format_data(self.Traindataname_truefeatures, self.Traindataname_truelabels, self.Testdataname_truefeatures, self.Testdataname_truelabels, self.pathwaytype)
        print(Trainfeas['numView'])
        Train_placeholders = get_placeholder(Trainfeas['adjs'], Trainfeas['numView'])
        Test_placeholders = get_placeholder(Testfeas['adjs'], Testfeas['numView'])
        model = ARGA(Train_placeholders, Trainfeas['numView'], Trainfeas['num_features'], label_number)
        opt = OptimizerAE(model=model, preds_fuze=model.reconstructions_fuze,
                          labels=Train_placeholders['adjs_orig'],
                          p=Train_placeholders['p'],
                          numView=Trainfeas['numView'],
                          pos_weights=Train_placeholders['pos_weights'],
                          fea_pos_weights=Train_placeholders['fea_pos_weights'],
                          norm=Train_placeholders['norm'],
                          true_labels=Train_placeholders['true_labels'],
                          predict_labels=model.embeddings2)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        for epoch in range(self.iterations):
            print("Train Epoch:", '%04d' % (epoch + 1))
            Train_prediction, reconstruct_loss = update(model, opt, sess, Trainfeas['adjs'], Trainfeas['adjs_label'], Trainfeas['features'], Train_placeholders, Trainfeas['pos_weights'], Trainfeas['fea_pos_weights'], Trainfeas['norms'], Trainfeas['true_labels'])
            print('reconstruct_loss', reconstruct_loss)
            Train_predict_labels = np.argmax(Train_prediction, axis=1)
            Trainfeas_true_labels = np.argmax(Trainfeas['true_labels'], axis=1)
            Testfeas_true_labels = np.argmax(Testfeas['true_labels'], axis=1)
            predicte_label_num = count_num(Train_predict_labels)
            true_label_num = count_num(Trainfeas_true_labels)
            print('predict label_num:', predicte_label_num)
            print('true label_num:', true_label_num)
            train_pm = predicting_metrics(Trainfeas_true_labels, Train_predict_labels, self.Traindataname, self.Testdataname)
            acc, f1_macro, precision_macro, recall_macro, mcc = train_pm.evaluationPredicteModelFromLabel()
            if epoch > 150 and (epoch + 1) % 5 == 0:
                graph = tf.get_default_graph()
                e_dense_1_0_vars = graph.get_tensor_by_name(
                    'arga/Encoder/e_dense_1_' + str(Trainfeas['numView']) + '_vars/weights:0')
                e_dense_1_0_vars = tf.identity(e_dense_1_0_vars)
                e_dense_2_0_vars = graph.get_tensor_by_name(
                    'arga/Encoder/e_dense_2_' + str(Trainfeas['numView']) + '_vars/weights:0')
                e_dense_2_0_vars = tf.identity(e_dense_2_0_vars)
                print(e_dense_1_0_vars)
                print(e_dense_2_0_vars)
                e_dense_3_0_vars = graph.get_tensor_by_name("arga/Encoder/e_dense_3_"+str(Trainfeas['numView'])+"/kernel:0")
                e_dense_3_0_vars = tf.identity(e_dense_3_0_vars)
                print(e_dense_3_0_vars)
                e_dense_3_0_biase = graph.get_tensor_by_name("arga/Encoder/e_dense_3_"+str(Trainfeas['numView'])+"/bias:0")
                e_dense_3_0_biase = tf.identity(e_dense_3_0_biase)
                print(e_dense_3_0_biase)
                x = Testfeas['features']
                x = tf.cast(x, dtype=tf.float32)
                x = tf.matmul(x, e_dense_1_0_vars)
                X = Testfeas['adjs'][int(Testfeas['numView'])-1]
                X = tf.cast(X, dtype=tf.float32)
                outputs = tf.matmul(X, x)
                outputs = tf.matmul(outputs, e_dense_2_0_vars)
                outputs = tf.matmul(X, outputs)
                outputs = tf.matmul(outputs, e_dense_3_0_vars) + e_dense_3_0_biase
                outputs = tf.nn.softmax(outputs)
                outputs_np = sess.run(outputs)
                Test_predict_labels = np.argmax(outputs_np, axis=1)
                Testfeas_true_labels = np.argmax(Testfeas['true_labels'], axis=1)
                test_pm = predicting_metrics(Testfeas_true_labels, Test_predict_labels, self.Traindataname, self.Testdataname)
                print('-----------------------test--------------------------------------------')
                acc, f1_macro, precision_macro, recall_macro, mcc = test_pm.TestevaluationPredicteModelFromLabel()
        # saver = tf.train.Saver()
        # saver.save(sess, "./Traindataname_"+Traindataname+"_Testdataname_"+Testdataname+"_model" + str(epoch)+"/MyModel")


        emb_ind2 = update_test(model, opt, sess, Trainfeas['adjs'], Trainfeas['adjs_label'], Trainfeas['features'], Train_placeholders,
                                Trainfeas['pos_weights'], Trainfeas['fea_pos_weights'], Trainfeas['norms'])
        kmeans = KMeans(n_clusters=label_number).fit(emb_ind2)
        y_pred_last = kmeans.labels_
        train_cm = predicting_metrics(Trainfeas_true_labels, y_pred_last, self.Traindataname, self.Testdataname)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, idx, mcc, nmi, adjscore = train_cm.evaluationClusterModelFromLabel()
        init_cluster = tf.constant(kmeans.cluster_centers_)
        print(init_cluster.shape)
        sess.run(tf.assign(model.cluster_layer.vars['clusters'], init_cluster))
        q = compute_q(model, opt, sess, Trainfeas['adjs'], Trainfeas['adjs_label'], Trainfeas['features'], Train_placeholders,
                      Trainfeas['pos_weights'], Trainfeas['fea_pos_weights'], Trainfeas['norms'])
        # print("q=")
        # print(q)
        p = target_distribution(q)

        for epoch in range(self.kl_iterations):
            emb, kl_loss = update_kl(model, opt, sess, Trainfeas['adjs'], Trainfeas['adjs_label'], Trainfeas['features'], p,
                                     Train_placeholders, Trainfeas['pos_weights'], Trainfeas['fea_pos_weights'], Trainfeas['norms'],  idx=idx, true_labels=Trainfeas['true_labels'])
            if epoch % 1 == 0:
                q = compute_q(model, opt, sess, Trainfeas['adjs'], Trainfeas['adjs_label'], Trainfeas['features'],
                              Train_placeholders, Trainfeas['pos_weights'], Trainfeas['fea_pos_weights'], Trainfeas['norms'])
                p = target_distribution(q)
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                # y_pred_last = y_pred
                print('delta_label', delta_label)
                print("Epoch:", '%04d' % (epoch + 1))
                kmeans = KMeans(n_clusters=label_number).fit(emb)
                y_pred_last = kmeans.labels_
                train_cm = predicting_metrics(Trainfeas_true_labels, y_pred_last, self.Traindataname, self.Testdataname)
                acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, idx, mcc, nmi, adjscore = train_cm.evaluationClusterModelFromLabel()


                prediction = update_test2(model, opt, sess, Trainfeas['adjs'], Trainfeas['adjs_label'],
                                       Trainfeas['features'], Train_placeholders,
                                       Trainfeas['pos_weights'], Trainfeas['fea_pos_weights'], Trainfeas['norms'])
                prediction = np.argmax(prediction, axis=1)
                train_pm = predicting_metrics(Trainfeas_true_labels, prediction, self.Traindataname,
                                              self.Testdataname)
                acc, f1_macro, precision_macro, recall_macro, mcc = train_pm.evaluationPredicteModelFromLabel()

                NMIs.append(nmi)
                loss.append(kl_loss)

                graph = tf.get_default_graph()
                e_dense_1_0_vars = graph.get_tensor_by_name(
                    'arga/Encoder/e_dense_1_' + str(Trainfeas['numView']) + '_vars/weights:0')
                e_dense_1_0_vars = tf.identity(e_dense_1_0_vars)
                e_dense_2_0_vars = graph.get_tensor_by_name(
                    'arga/Encoder/e_dense_2_' + str(Trainfeas['numView']) + '_vars/weights:0')
                e_dense_2_0_vars = tf.identity(e_dense_2_0_vars)
                print(e_dense_1_0_vars)
                print(e_dense_2_0_vars)
                e_dense_3_0_vars = graph.get_tensor_by_name(
                    "arga/Encoder/e_dense_3_" + str(Trainfeas['numView']) + "/kernel:0")
                e_dense_3_0_vars = tf.identity(e_dense_3_0_vars)
                print(e_dense_3_0_vars)
                e_dense_3_0_biase = graph.get_tensor_by_name(
                    "arga/Encoder/e_dense_3_" + str(Trainfeas['numView']) + "/bias:0")
                e_dense_3_0_biase = tf.identity(e_dense_3_0_biase)
                print(e_dense_3_0_biase)
                x = Testfeas['features']
                x = tf.cast(x, dtype=tf.float32)
                x = tf.matmul(x, e_dense_1_0_vars)
                X = Testfeas['adjs'][int(Testfeas['numView']) - 1]
                X = tf.cast(X, dtype=tf.float32)
                outputs = tf.matmul(X, x)
                outputs = tf.matmul(outputs, e_dense_2_0_vars)
                embedding = tf.matmul(X, outputs)
                outputs = tf.matmul(embedding, e_dense_3_0_vars) + e_dense_3_0_biase
                outputs = tf.nn.softmax(outputs)
                # embedding_np = sess.run(embedding)
                # kmeans2 = KMeans(n_clusters=label_number).fit(embedding_np)
                # y_pred_last2 = kmeans2.labels_
                # test_cm = predicting_metrics(Testfeas_true_labels, y_pred_last2, self.Traindataname,
                #                              self.Testdataname)
                # acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, idx, mcc, nmi, adjscore = test_cm.evaluationClusterModelFromLabel2()
                outputs_np = sess.run(outputs)
                Test_predict_labels = np.argmax(outputs_np, axis=1)
                Testfeas_true_labels = np.argmax(Testfeas['true_labels'], axis=1)
                test_pm = predicting_metrics(Testfeas_true_labels, Test_predict_labels, self.Traindataname,
                                             self.Testdataname)
                print('-----------------------test--------------------------------------------')
                acc, f1_macro, precision_macro, recall_macro, mcc = test_pm.TestevaluationPredicteModelFromLabel()

                if epoch > 0 and delta_label < self.tol:
                    graph = tf.get_default_graph()
                    e_dense_1_0_vars = graph.get_tensor_by_name(
                        'arga/Encoder/e_dense_1_' + str(Trainfeas['numView']) + '_vars/weights:0')
                    e_dense_1_0_vars = tf.identity(e_dense_1_0_vars)
                    e_dense_2_0_vars = graph.get_tensor_by_name(
                        'arga/Encoder/e_dense_2_' + str(Trainfeas['numView']) + '_vars/weights:0')
                    e_dense_2_0_vars = tf.identity(e_dense_2_0_vars)
                    print(e_dense_1_0_vars)
                    print(e_dense_2_0_vars)
                    e_dense_3_0_vars = graph.get_tensor_by_name(
                        "arga/Encoder/e_dense_3_" + str(Trainfeas['numView']) + "/kernel:0")
                    e_dense_3_0_vars = tf.identity(e_dense_3_0_vars)
                    print(e_dense_3_0_vars)
                    e_dense_3_0_biase = graph.get_tensor_by_name(
                        "arga/Encoder/e_dense_3_" + str(Trainfeas['numView']) + "/bias:0")
                    e_dense_3_0_biase = tf.identity(e_dense_3_0_biase)
                    print(e_dense_3_0_biase)
                    x = Testfeas['features']
                    x = tf.cast(x, dtype=tf.float32)
                    x = tf.matmul(x, e_dense_1_0_vars)
                    X = Testfeas['adjs'][int(Testfeas['numView']) - 1]
                    X = tf.cast(X, dtype=tf.float32)
                    outputs = tf.matmul(X, x)
                    outputs = tf.matmul(outputs, e_dense_2_0_vars)
                    embedding = tf.matmul(X, outputs)
                    outputs = tf.matmul(embedding, e_dense_3_0_vars) + e_dense_3_0_biase
                    outputs = tf.nn.softmax(outputs)
                    # embedding_np = sess.run(embedding)
                    # kmeans2 = KMeans(n_clusters=label_number).fit(embedding_np)
                    # y_pred_last2 = kmeans2.labels_
                    # test_cm = predicting_metrics(Testfeas_true_labels, y_pred_last2, self.Traindataname,
                    #                              self.Testdataname)
                    # acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, idx, mcc, nmi, adjscore = test_cm.evaluationClusterModelFromLabel2()
                    outputs_np = sess.run(outputs)
                    Test_predict_labels = np.argmax(outputs_np, axis=1)
                    Testfeas_true_labels = np.argmax(Testfeas['true_labels'], axis=1)
                    test_pm = predicting_metrics(Testfeas_true_labels, Test_predict_labels, self.Traindataname,
                                                 self.Testdataname)
                    print('-----------------------test--------------------------------------------')
                    acc, f1_macro, precision_macro, recall_macro, mcc = test_pm.TestevaluationPredicteModelFromLabel()
                    print("early_stop")
                    break

        print('NMI', NMIs)
        print('loss', loss)
        graph = tf.get_default_graph()
        e_dense_1_0_vars = graph.get_tensor_by_name(
            'arga/Encoder/e_dense_1_' + str(Trainfeas['numView']) + '_vars/weights:0')
        e_dense_1_0_vars = tf.identity(e_dense_1_0_vars)
        e_dense_2_0_vars = graph.get_tensor_by_name(
            'arga/Encoder/e_dense_2_' + str(Trainfeas['numView']) + '_vars/weights:0')
        e_dense_2_0_vars = tf.identity(e_dense_2_0_vars)
        print(e_dense_1_0_vars)
        print(e_dense_2_0_vars)
        e_dense_3_0_vars = graph.get_tensor_by_name(
            "arga/Encoder/e_dense_3_" + str(Trainfeas['numView']) + "/kernel:0")
        e_dense_3_0_vars = tf.identity(e_dense_3_0_vars)
        print(e_dense_3_0_vars)
        e_dense_3_0_biase = graph.get_tensor_by_name(
            "arga/Encoder/e_dense_3_" + str(Trainfeas['numView']) + "/bias:0")
        e_dense_3_0_biase = tf.identity(e_dense_3_0_biase)
        print(e_dense_3_0_biase)
        x = Testfeas['features']
        x = tf.cast(x, dtype=tf.float32)
        x = tf.matmul(x, e_dense_1_0_vars)
        X = Testfeas['adjs'][int(Testfeas['numView']) - 1]
        X = tf.cast(X, dtype=tf.float32)
        outputs = tf.matmul(X, x)
        outputs = tf.matmul(outputs, e_dense_2_0_vars)
        embedding = tf.matmul(X, outputs)
        outputs = tf.matmul(embedding, e_dense_3_0_vars) + e_dense_3_0_biase
        outputs = tf.nn.softmax(outputs)
        # embedding_np = sess.run(embedding)
        # kmeans2 = KMeans(n_clusters=label_number).fit(embedding_np)
        # y_pred_last2 = kmeans2.labels_
        # test_cm = predicting_metrics(Testfeas_true_labels, y_pred_last2, self.Traindataname, self.Testdataname)
        # acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, idx, mcc, nmi, adjscore = test_cm.evaluationClusterModelFromLabel2()
        outputs_np = sess.run(outputs)
        Test_predict_labels = np.argmax(outputs_np, axis=1)
        Testfeas_true_labels = np.argmax(Testfeas['true_labels'], axis=1)
        test_pm = predicting_metrics(Testfeas_true_labels, Test_predict_labels, self.Traindataname,
                                     self.Testdataname)
        print('-----------------------test--------------------------------------------')
        acc, f1_macro, precision_macro, recall_macro, mcc = test_pm.TestevaluationPredicteModelFromLabel()
        # saver2 = tf.train.Saver()
        # saver2.save(sess, "./have_KL-Traindataname_"+Traindataname+"_Testdataname_"+Testdataname+"_model" + str(epoch)+"/MyModel")
        return acc, f1_macro, precision_macro, nmi, adjscore

if __name__ == '__main__':
    iterations = 200
    kl_iterations = 40
    times = 1
    tol = 0.1



    Traindataname = 'HBone116d'
    Testdataname = 'HBone118d'
    Traindataname_truefeatures = "C:/Users/User/Desktop/data/hBone/116/GSM4626770_fin_116_disease.csv"
    Traindataname_truelabels = 'C:/Users/User/Desktop/data/hBone/116/GSM4626770_Label.csv'
    Testdataname_truefeatures = 'C:/Users/User/Desktop/data/hBone/118/GSM4626771_fin_118_disease.csv'
    Testdataname_truelabels = 'C:/Users/User/Desktop/data/hBone/118/GSM4626771_Label.csv'
    pathwaytype = 'C:/Users/User/Desktop/pathway/cross_omics'
    settings = get_settings(Traindataname, Traindataname_truefeatures, Traindataname_truelabels, Testdataname, Testdataname_truefeatures, Testdataname_truelabels, iterations, kl_iterations, tol, pathwaytype)
    runner = Clustering_Runner(settings)
    acc, f1_macro, precision_macro, recall_macro, mcc = runner.erun()
    print(acc, f1_macro, precision_macro, recall_macro, mcc)





