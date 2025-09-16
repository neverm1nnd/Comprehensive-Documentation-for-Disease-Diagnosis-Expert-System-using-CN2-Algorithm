import numpy as np

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        true_idx = np.where(labels == y_true[i])[0][0]
        pred_idx = np.where(labels == y_pred[i])[0][0]
        cm[true_idx, pred_idx] += 1
    return cm

def classification_metrics(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels)
    n_classes = len(labels)
    
    # Обчислення TP, FP, FN, TN для кожного класу
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    
    # Accuracy
    accuracy = TP.sum() / cm.sum()
    
    # Precision, Recall, F1-score для кожного класу
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)
    for i in range(n_classes):
        if TP[i] + FP[i] > 0:
            precision[i] = TP[i] / (TP[i] + FP[i])
        if TP[i] + FN[i] > 0:
            recall[i] = TP[i] / (TP[i] + FN[i])
        if precision[i] + recall[i] > 0:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    
    # Макро-середні значення
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_score.mean()
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
