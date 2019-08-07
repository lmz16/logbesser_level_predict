import numpy as np
from sklearn import metrics

def ROC(labels, pred, label_kinds):

    fpr = []
    tpr = []

    for threshold in np.arange(0, 1.01, 0.01):

        temp_tp = np.zeros((1, label_kinds))
        temp_fp = np.zeros((1, label_kinds))
        temp_fn = np.zeros((1, label_kinds))
        temp_tn = np.zeros((1, label_kinds))
        temp_fpr = np.zeros((1, label_kinds))
        temp_tpr = np.zeros((1, label_kinds))

        for kind in range(label_kinds):

            for i, label in enumerate(labels):

                vector = pred[i]

                if vector[kind] >= threshold and kind == label:

                    temp_tp[0, kind] += 1

                if vector[kind] >= threshold and kind != label:

                    temp_fp[0, kind] += 1

                if vector[kind] < threshold and kind == label:

                    temp_fn[0, kind] += 1

                if vector[kind] < threshold and kind != label:

                    temp_tn[0, kind] += 1

            if temp_tp[0, kind] + temp_fn[0, kind] == 0:
            
                temp_tpr[0, kind] = 1
                
            else:
            
                temp_tpr[0, kind] = temp_tp[0, kind] / (temp_tp[0, kind] + temp_fn[0, kind])
                
            if temp_tp[0, kind] + temp_fn[0, kind] == 0:
            
                temp_fpr[0, kind] = 0
                
            else:
            
                temp_fpr[0, kind] = temp_fp[0, kind] / (temp_fp[0, kind] + temp_tn[0, kind])

        fpr.append(np.average(temp_fpr))
        tpr.append(np.average(temp_tpr))

    return fpr, tpr
    

def AUC(labels, pred, label_kinds):

    fpr, tpr = ROC(labels, pred, label_kinds)

    return metrics.auc(fpr, tpr)
    
    
def BrierScore(labels, pred, label_kinds):

    score = 0

    for i, label in enumerate(labels):
    
        for j in range(label_kinds):
        
            score += (pred[i][j] - int(j==label))**2
            
    score /= (label_kinds * len(pred))
    
    return score