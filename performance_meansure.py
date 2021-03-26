from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, roc_curve, auc
import pylab as pl

def my_metrics(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted')
    f1Score=f1_score(y_true, y_pred, average='weighted')
    #fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=2)
    #auc_score = auc(fpr, tpr)
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    #print("AUC : {}".format(auc_score))
    cm=confusion_matrix(y_true, y_pred)
    print(cm)
    return accuracy, precision, f1Score

def plot_results(results):
    pl.figure()

    pl.subplot(121)
    pl.plot(results['train_acc'])
    pl.title('Accuracy:')
    pl.plot(results['test_acc'])
    pl.legend(('Train', 'Test'))

    pl.subplot(122)
    pl.plot(results['train_loss'])
    pl.title('Cost:')
    pl.plot(results['test_loss'])
    pl.legend(('Train', 'Test'))