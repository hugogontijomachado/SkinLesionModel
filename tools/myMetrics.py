from sklearn.metrics import *
import numpy as np

def fromInterprets(interprets):
    """
    PS: As métricas ainda precisam ser revisadas!!!
    """
    if not isinstance(interprets,list):
        print('oi')
        interprets = [interprets]
    metrics = []
    for i, interp in enumerate(interprets):
        p = [int(p.argmax()) for p in interp.preds]
        t = [int(t) for t in interp.targs]
        metrics.append({
            "Model": {i},
            "F1 score": f1_score(t, p, average='weighted'),
            "Balanced Accuracy": balanced_accuracy_score(t ,p),
            "Accuracy": accuracy_score(t, p),
            "ROC_AUC Score": roc_auc_score(t, p, average='weighted'),
            "Precision": precision_score(t, p, average='weighted'),
            "Average Precision": average_precision_score(t, p, average='weighted'),
        })
    return metrics

def printMeanMetrics(metrics):
    metricValues = np.array([np.array(list(metric.values())) for metric in metrics])
    metricKeys = list(metrics[0].keys())
    print( "Mean metrics for all models")
    for m, key in zip(metricValues[:,1:].mean(0),metricKeys[1:]):
        print(f'{key}: {m}')
    #return metricValues, metricKeys

def printMetrics(metrics, n=None):
    metric = metrics[n] if n else metrics
    for key, value in metric.items():
        print(f'{key}= {value}')

    