from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Evaluation:")
    print(classification_report(y_test, y_pred))

def get_confusion(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)

def get_roc_data(model, X_test, y_test):
    probas = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probas)
    return fpr, tpr, auc(fpr, tpr)
