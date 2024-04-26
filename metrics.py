from sklearn import metrics
from munkres import Munkres, print_matrix
import numpy as np
class predicting_metrics():
    def __init__(self, true_label, predict_label, Trainname, Testname):
        self.true_label = true_label
        self.pred_label = predict_label
        self.Trainname = Trainname
        self.Testname = Testname
    def predictingAcc(self):
        Trainname = self.Trainname
        Testname = self.Testname
        acc = metrics.accuracy_score(self.true_label, self.pred_label)
        f1_macro = metrics.f1_score(self.true_label, self.pred_label, average='macro')
        precision_macro = metrics.precision_score(self.true_label, self.pred_label, average='macro')
        recall_macro = metrics.recall_score(self.true_label, self.pred_label, average='macro')
        f1_micro = metrics.f1_score(self.true_label, self.pred_label, average='micro')
        precision_micro = metrics.precision_score(self.true_label, self.pred_label, average='micro')
        recall_micro = metrics.recall_score(self.true_label, self.pred_label, average='micro')
        mcc = metrics.matthews_corrcoef(self.true_label, self.pred_label)
        return Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc
    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)
        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0
        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]
                cost[i][j] = len(mps_d)
        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)
        idx = 0
        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c
        Trainname = self.Trainname
        Testname = self.Testname
        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        mcc = metrics.matthews_corrcoef(self.true_label, self.pred_label)
        return Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc, idx

    def evaluationPredicteModelFromLabel(self):
        Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc = self.predictingAcc()
        print('Train----Traindataname: %s, Testdataname: %s, ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, mcc=%f' % (Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc))
        fh = open('Traindataname_%s_Testdataname_%s_recoder.txt' % (Trainname, Testname), 'a')
        fh.write('Train----Traindataname: %s, Testdataname: %s, ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, mcc=%f' % (Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc))
        fh.write('\r\n')
        fh.flush()
        fh.close()
        return acc, f1_macro, precision_macro, recall_macro, mcc
    def TestevaluationPredicteModelFromLabel(self):
        Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc = self.predictingAcc()
        print('Test----Traindataname: %s, Testdataname: %s, ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, mcc=%f' % (Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc))
        fh = open('Traindataname_%s_Testdataname_%s_recoder.txt' % (Trainname, Testname), 'a')
        fh.write('Test----Traindataname: %s, Testdataname: %s, ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, mcc=%f' % (Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc))
        fh.write('\r\n')
        fh.flush()
        fh.close()
        return acc, f1_macro, precision_macro, recall_macro, mcc
    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc, idx = self.clusteringAcc()
        print('KL-Cluster--Train--Traindataname: %s, Testdataname: %s, ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, mcc=%f, nmi=%f, adjscore=%f' % (Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc, nmi, adjscore))
        fh = open('Traindataname_%s_Testdataname_%s_recoder.txt' % (Trainname, Testname), 'a')
        fh.write('KL-Cluster--Train--Traindataname: %s, Testdataname: %s, ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, mcc=%f, nmi=%f, adjscore=%f' % (Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc, nmi, adjscore))
        fh.write('\r\n')
        fh.flush()
        fh.close()
        return  acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc, idx, nmi, adjscore

    def evaluationClusterModelFromLabel2(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc, idx = self.clusteringAcc()
        print('KL-Cluster--Test--Traindataname: %s, Testdataname: %s, ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, mcc=%f, nmi=%f, adjscore=%f' % (Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc, nmi, adjscore))
        fh = open('Traindataname_%s_Testdataname_%s_recoder.txt' % (Trainname, Testname), 'a')
        fh.write('KL-Cluster--Test--Traindataname: %s, Testdataname: %s, ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, mcc=%f, nmi=%f, adjscore=%f' % (Trainname, Testname, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc, nmi, adjscore))
        fh.write('\r\n')
        fh.flush()
        fh.close()
        return  acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, mcc, idx, nmi, adjscore