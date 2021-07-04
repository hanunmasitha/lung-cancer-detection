from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, confusion_matrix, roc_curve, auc
import pylab as pl

def my_metrics(y_true, y_pred, classes, predictions):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    # f1Score=f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(classes, predictions, average='macro', multi_class='ovr')
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("AUC : {}".format(roc_auc))
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return accuracy, precision, roc_auc

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