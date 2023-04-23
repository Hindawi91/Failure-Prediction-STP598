import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
from itertools import cycle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from glob import glob                                                           
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix,roc_auc_score
import timeit
import datetime

All_Models = []
All_Metrics = []
All_CMs = []

def balanced_error_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    FPR = fp/(fp+tn)
    FNR = fn/(fn+tp)
    BER = 0.5 * (FPR + FNR)
    return BER

seed = 520
np.set_printoptions(precision=3)

myfile = r"./Data2.csv"

mydata = pd.read_csv(myfile, header=0)
mydata.drop(mydata.iloc[:, -7::], inplace = True, axis = 1)
mydata['Failure'].replace({"Yes":1,"No":0}, inplace=True)

X_new = mydata[['Hours Since Previous Failure','HoursSinceStart','Measure8',
         'Measure10','Measure9','Measure11','Humidity','Temperature']]

y = mydata['Failure']


y = y.to_numpy()
X_new = X_new.to_numpy()

scaler = StandardScaler()
scaler.fit(X_new)
X_new_transformed = scaler.transform(X_new)

X = X_new_transformed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42,stratify=y)

print ("number of failures in train set = ",np.count_nonzero(y_train == 0))
print ("number of failures in test set = ",np.count_nonzero(y_test == 0))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)





Gradient_Boost = GradientBoostingClassifier(learning_rate=0.01, max_depth = 100, min_samples_split=5,
                                            n_estimators=200, min_samples_leaf =1, max_features = 'sqrt')

Neural_Network = MLPClassifier((200, 200),activation = 'relu',alpha = 10.0)

SVM = SVC(class_weight='balanced',probability=True,C=0.001, degree=2, kernel='linear')

Forrest = RandomForestClassifier(n_estimators=2000, max_depth=50,bootstrap =False,
                                 min_samples_split=10, min_samples_leaf=2,max_features='auto')


tree = DecisionTreeClassifier(criterion='entropy',max_features='auto', splitter='best')
AdaBoost_tree = AdaBoostClassifier(base_estimator = tree, n_estimators=400,learning_rate=1e-05, algorithm = 'SAMME')


                                           

voter1 = VotingClassifier(estimators = [("SVM",SVM),("Neural_Network",Neural_Network),
                          ("AdaBoost_tree",AdaBoost_tree),("Forrest",Forrest),
                          ("Gradient_Boost",Gradient_Boost)],voting = 'hard')

# voter2 = VotingClassifier(estimators = [("SVM",SVM),("Neural_Network",Neural_Network),
#                           ("AdaBoost_tree",AdaBoost_tree),("Forrest",Forrest),
#                           ("Gradient_Boost",Gradient_Boost)],voting = 'soft')

voter3 = VotingClassifier(estimators = [("SVM",SVM),
                          ("AdaBoost_tree",AdaBoost_tree),("Forrest",Forrest),
                          ("Gradient_Boost",Gradient_Boost)],voting = 'hard')

# voter4 = VotingClassifier(estimators = [("SVM",SVM),
#                           ("AdaBoost_tree",AdaBoost_tree),("Forrest",Forrest),
#                           ("Gradient_Boost",Gradient_Boost)],voting = 'soft')

voter5 = VotingClassifier(estimators = [("Neural_Network",Neural_Network),
                          ("AdaBoost_tree",AdaBoost_tree),("Forrest",Forrest),
                          ("Gradient_Boost",Gradient_Boost)],voting = 'hard')


# voter6 = VotingClassifier(estimators = [("Neural_Network",Neural_Network),
#                           ("AdaBoost_tree",AdaBoost_tree),("Forrest",Forrest),
#                           ("Gradient_Boost",Gradient_Boost)],voting = 'soft')



# Create StratifiedKFold object.
# models = [voter1,voter2,voter3,voter4,voter5,voter6]
models = [voter1,voter3,voter5]
for i,model in enumerate(models):
    
    imba_pipeline = make_pipeline(SMOTE(random_state=42),model)   
    score = cross_val_score(imba_pipeline, X_train, y_train, scoring='balanced_accuracy', cv=kf,n_jobs=-1)

    print(f"Voter{i} score = {score}")

    model.fit(X_train,y_train)
    
    # print('Best parameters found:\n', model.best_params_)
    # print('Best CV Balanced Accuracy', model.best_score_)
    
    y_pred = model.predict(X_test)
    # y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_true = y_test
    
    print('Balanced Accuracy on test data:')
    print(metrics.balanced_accuracy_score(y_true, y_pred) )
    
    print("\n")
    
    # print('roc_auc_ovr_weighted on test data:')
    # print(metrics.roc_auc_score(y_true, y_pred_prob, average='weighted') )
    
    print("\n")
    
    print('Balanced Error on test data:')
    print(balanced_error_rate(y_true, y_pred))
    
    print("\n")
    
    print('Classification report on test data: ')
    print(classification_report(y_true, y_pred))
    
    print('Confusion Matrix on test data: ')
    print(confusion_matrix(y_true, y_pred))
    
    
    def firas_tests(model_name,y_true, y_pred):
            
        acc = accuracy_score(y_true, y_pred)
        
        f1_none = f1_score(y_true, y_pred, average=None)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        precision_none = precision_score(y_true, y_pred, average=None)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        
        recall_none = recall_score(y_true, y_pred, average=None)
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        
        # ROC_AUC_ovr = roc_auc_score(y_true, y_pred_prob,multi_class= 'ovr')
        # ROC_AUC_ovo = roc_auc_score(y_true, y_pred_prob,multi_class= 'ovo')
        
        Balanced_acc = balanced_accuracy_score(y_true, y_pred)
        Balanced_error = balanced_error_rate(y_true, y_pred)
        # Best_parameters = search_1.best_params_
        
        CM = confusion_matrix(y_true, y_pred)
        
        metrics = [model_name,acc,f1_none,f1_macro,f1_micro,f1_weighted,precision_none,
                    precision_macro,precision_micro,precision_weighted,recall_none,
                    recall_macro,recall_micro,recall_weighted,
                    Balanced_acc,Balanced_error]
        metrics_names = ["model_name","Accuracy","F1_none","F1_macro",
                          "F1_micro","F1_weighted","Precision_none",
                          "Precision_macro","Precision_micro",
                          "Precision_weighted","Recall_none",
                          "Recall_macro","Recall_micro",
                          "Recall_weighted",
                          "Balanced_acc","Balanced_error"]
            
    #         ALL_Models.appeand(model_name)
        print (CM)
        All_Metrics.append(metrics)
        All_CMs.append(CM)
    
    
    
        #Send Metrics To Excel Sheet
    
        df = pd.DataFrame(All_Metrics,columns=metrics_names)
        
        df.to_excel (f'{model_name} Metrics.xlsx', index = False, header=True)
        
        print(df.shape)
    
    
        df = pd.DataFrame([All_CMs])
        
        df.to_excel (f'{model_name} CMs.xlsx', index = False, header=True)
        
        print(df.shape)
    
    
    
        frames = []
        
        for cm in All_CMs:
            df = pd.DataFrame(cm)
            frames.append(df)
        
        final = pd.concat(frames)
        frames = []
        
        for cm in All_CMs:
            df = pd.DataFrame(cm)
            frames.append(df)
        
        final = pd.concat(frames)
        
        final.to_excel (f'{model_name} CMs2.xlsx', index = False, header=True)
    
    
    def make_ROC(model_name):
        probs = model.predict_proba(X_test)
        preds = probs[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        
        # method I: plt
        
        plt.title('Receiver Operating Characteristic')
        plt.suptitle(f'{model_name}')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.show()
        
        
        plt.savefig(f'{model_name} ROC.png',figsize=(20,10))
        
    # make_ROC(f"voter{i}")    
    firas_tests(f"voter{i}",y_true, y_pred)



# calculate the fpr and tpr for all thresholds of the classification












