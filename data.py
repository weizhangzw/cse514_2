import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV
import tensorflow
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA 


X = np.loadtxt('a_c.data', dtype='int', delimiter=',', usecols=range(1, 16))
y = np.loadtxt('a_c.data', dtype='str', delimiter=',', usecols=[0])

X_train, X_val, tp_y_train, tp_y_val = train_test_split(X, y, test_size=0.1, random_state=42)
y_train = tp_y_train.reshape(-1, 1)
y_val = tp_y_val.reshape(-1, 1)

y_train_onehot = np.array(pd.get_dummies(tp_y_train))
y_val_onehot = np.array(pd.get_dummies(tp_y_val))

def accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return np.mean(y_pred == y_test)

fold_num = 5



# dimension reduction
# pca_model = PCA(n_components=4,svd_solver='full',random_state=30,whiten=False,tol=0.5)
# pca_model.fit(X_train)
# X_train = pca_model.transform(X_train)
# X_val = pca_model.transform(X_val)
# from sklearn.neighbors import KNeighborsClassifier
# print('--------------------------------------------------------------------------------------')
# print('Model: KNN')
# k_scores = []
# k_range = range(1, 31)
# start = time.time()
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k,metric= 'chebyshev',algorithm='ball_tree',leaf_size=30,weights='distance',p=4)
#     kf = KFold(n_splits=fold_num)
#     acc = 0
#     for i, (train_index, test_index) in enumerate(kf.split(X_train)):
#         X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
#         y_train_fold, y_test_fold = tp_y_train[train_index], tp_y_train[test_index]
        

#         knn.fit(X_train_fold, y_train_fold)
#         acc += accuracy(knn, X_test_fold, y_test_fold)
#     acc /= fold_num
#     k_scores.append(acc)
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Accuracy')
# plt.title('Cross validation result of KNN(A,C)')
# plt.show()
# end = time.time()
# print(end-start)


# dimension reduction
# pca_model = PCA(n_components=4,svd_solver='full',random_state=30,whiten=False,tol=0.5)
# pca_model.fit(X_train)
# X_train = pca_model.transform(X_train)
# X_val = pca_model.transform(X_val)
# print('--------------------------------------------------------------------------------------')
# print('Model: Decision Tree')
# from sklearn.tree import DecisionTreeClassifier
# k_scores = []
# k_range = range(1, 31)
# st = time.time()
# for k in k_range:
#     df = DecisionTreeClassifier(random_state=0, max_depth=k,splitter='random', criterion='gini',min_weight_fraction_leaf=0.01)
#     kf = KFold(n_splits=fold_num)
#     acc = 0
    
#     for i, (train_index, test_index) in enumerate(kf.split(X_train)):
#         X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
#         y_train_fold, y_test_fold = tp_y_train[train_index], tp_y_train[test_index]
#         df.fit(X_train_fold, y_train_fold)
#         acc += accuracy(df, X_test_fold, y_test_fold)
#     acc /= fold_num
#     k_scores.append(acc)

# plt.plot(k_range, k_scores)
# plt.xlabel('Value of max_depth')
# plt.ylabel('Accuracy')
# plt.title('Cross validation result of Decision Tree(A,C)')
# plt.show()
# ed = time.time()
# print(ed-st)


# dimension reduction
# nmf = make_pipeline(StandardScaler(), NMF(n_components=3))
# nmf_model = NMF(n_components=4,init='nndsvdar',solver='mu',max_iter=150,alpha=0.02)
# nmf_model.fit(X_train)
# X_train = nmf_model.transform(X_train)
# X_val = nmf_model.transform(X_val)
# print('--------------------------------------------------------------------------------------')
# print('Model: Random Forest')
# from sklearn.ensemble import RandomForestClassifier
# k_scores = []
# k_range = range(1, 31)
# st = time.time()
# for k in k_range:
#     rf = RandomForestClassifier(random_state=0, max_depth=k,max_features='sqrt',criterion='gini',min_impurity_decrease=0.02,bootstrap=True)
#     kf = KFold(n_splits=fold_num)
#     acc = 0
    
#     for i, (train_index, test_index) in enumerate(kf.split(X_train)):
#         X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
#         y_train_fold, y_test_fold = tp_y_train[train_index], tp_y_train[test_index]
#         rf.fit(X_train_fold, y_train_fold)
#         acc += accuracy(rf, X_test_fold, y_test_fold)
#     acc /= fold_num
#     k_scores.append(acc)

# plt.plot(k_range, k_scores)
# plt.xlabel('Value of max_depth in Random Forest')
# plt.ylabel('Accuracy')
# plt.title('Cross validation result of Random Forest(M,Y)')
# ed=time.time()
# print(ed-st)
# plt.show()


# dimension reduction
# fa = make_pipeline(StandardScaler(), FastICA(n_components=3))
# fa_model = FastICA(n_components=4,algorithm='parallel',max_iter=150,random_state=30,whiten=True)
# fa_model.fit(X_train)
# X_train = fa_model.transform(X_train)
# X_val = fa_model.transform(X_val)
# print('--------------------------------------------------------------------------------------')
# print('Model: SVM')
# from sklearn.svm import SVC
# k_scores = []
# k_range = range(1, 31)
# st = time.time()
# for k in k_range:
#     svc = SVC(kernel='poly', degree=k,cache_size=300,shrinking=False,gamma='scale',coef0=0.02,probability=False)
#     kf = KFold(n_splits=fold_num)
#     acc = 0
    
#     for i, (train_index, test_index) in enumerate(kf.split(X_train)):
#         X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
#         y_train_fold, y_test_fold = tp_y_train[train_index], tp_y_train[test_index]
#         svc.fit(X_train_fold, y_train_fold)
#         acc += accuracy(svc, X_test_fold, y_test_fold)
#     acc /= fold_num
#     k_scores.append(acc)

# plt.plot(k_range, k_scores)
# plt.xlabel('Value of max_depth in SVM')
# plt.ylabel('Accuracy')
# plt.title('Cross validation result of SVM(A,C)')
# plt.show()
# ed=time.time()
# print(ed-st)


# from sklearn.decomposition import SparsePCA
# spca_model = SparsePCA(n_components=4,random_state=30,tol=0.5)
# spca_model.fit(X_train)
# X_train = spca_model.transform(X_train)
# X_val = spca_model.transform(X_val)
# print('--------------------------------------------------------------------------------------')
# print('Model: ANN')
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# st = time.time()
# node_per_layer = np.arange(2,31,1)
# accuracies = np.zeros(len(node_per_layer))
# for i, input in enumerate(node_per_layer):
#     kf = KFold(n_splits=fold_num)
#     acc = 0
#     for train_index, test_index in kf.split(X_train):
#         classifier = keras.Sequential(name="sigmoid_model")
#         classifier.add(layers.Dense(input, activation="relu"))
#         classifier.add(layers.Dense(input, activation="sigmoid"))
#         classifier.add(layers.Dense(input, activation="sigmoid"))
#         classifier.add(layers.Dense(2, activation="softmax"))

#         classifier.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        
#         X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
#         y_train_fold, y_test_fold = y_train_onehot[train_index], y_train_onehot[test_index]
#         classifier.fit(X_train_fold, y_train_fold, epochs = 25,batch_size = 10) 

#         val_loss, val_acc = classifier.evaluate(X_test_fold, y_test_fold, verbose=2)
#         acc += val_acc
#     acc /= fold_num
#     accuracies[i] = acc
# # Plot accuracy
# plt.plot(node_per_layer, accuracies)
# plt.xlabel('Node per layer count')
# plt.ylabel('Accuracy')
# plt.title('Cross validation result of ANN(A,C)')
# plt.show()
# ed=time.time()
# print(ed-st)


# dimension reduction
# from sklearn.decomposition import KernelPCA
# kernel_pca = KernelPCA(n_components=4, kernel="poly", fit_inverse_transform=True, alpha=1,random_state=30)
# kernel_pca.fit(X_train)
# X_train = kernel_pca.transform(X_train)
# X_val = kernel_pca.transform(X_val)
# print('--------------------------------------------------------------------------------------')
# print('Model: AdaBoost')
# from sklearn.ensemble import AdaBoostClassifier
# k_scores = []
# k_range = range(1, 31)
# st = time.time()
# for k in k_range:
#     ada = AdaBoostClassifier(n_estimators=k,learning_rate=0.5,algorithm='SAMME',random_state=20,base_estimator=None)
#     kf = KFold(n_splits=fold_num)
#     acc = 0
    
#     for i, (train_index, test_index) in enumerate(kf.split(X_train)):
#         X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
#         y_train_fold, y_test_fold = tp_y_train[train_index], tp_y_train[test_index]
#         ada.fit(X_train_fold, y_train_fold)
#         acc += accuracy(ada, X_test_fold, y_test_fold)
#     acc /= fold_num
#     k_scores.append(acc)

# plt.plot(k_range, k_scores)
# plt.xlabel('Number of estimators')
# plt.ylabel('Accuracy')
# plt.title('Cross validation result of AdaBoost(A,C)')
# plt.show()


#dimension reduction 
# from sklearn.manifold import MDS
# mds = MDS(n_components=4, random_state=30,verbose=0.2,dissimilarity='euclidean')
# X_train =mds.fit_transform(X_train)
# X_val = mds.fit_transform(X_val)
# print('--------------------------------------------------------------------------------------')
# from sklearn.linear_model import RidgeClassifier
# k_scores = []
# k_range = range(1, 31)
# st = time.time()
# for k in k_range:
#     rc = RidgeClassifier(alpha=k,solver='saga',random_state=30,fit_intercept=True,normalize=True)
#     kf = KFold(n_splits=fold_num)
#     acc = 0
    
#     for i, (train_index, test_index) in enumerate(kf.split(X_train)):
#         X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
#         y_train_fold, y_test_fold = tp_y_train[train_index], tp_y_train[test_index]
#         rc.fit(X_train_fold, y_train_fold)
#         acc += accuracy(rc, X_test_fold, y_test_fold)
#     acc /= fold_num
#     k_scores.append(acc)

# plt.plot(k_range, k_scores)
# plt.xlabel('Range')
# plt.ylabel('Accuracy')
# plt.title('Cross validation result of Ridge Classifier(H,K)')
# plt.show()

