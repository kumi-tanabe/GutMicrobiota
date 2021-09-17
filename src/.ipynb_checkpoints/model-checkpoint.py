import os
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Activation, GlobalAveragePooling2D
from keras import optimizers
from keras.utils import np_utils
from keras.applications.xception import Xception

from sklearn.metrics import confusion_matrix as skl_confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

sample_name = ['HF0','HF5',
               'DSS0','DSS1','DSS2','DSS3','DSS4','DSS5','DSS6','DSS7','DSS8','DSS9',
               'ADSS1','ADSS2','ADSS3','ADSS7']

metagenome_name = ['Bacteroides', 'Prevotella', 'Rikenellaceae', 'S24-7',
                   'Lactobacillus', 'Turicibacter', 'Clostridiales', 'Clostridiaceae',
                   'Lachnospiraceae', 'Ruminococcaceae', 'Oscillospira', 'Erysipelotrichaceae',
                   'Phyllobacteriaceae', 'Sphingomonadaceae', 'Ralstonia', 'Others']
    
nb_classes = 16
img_rows, img_cols, img_channels = 256, 256, 3

batch_size = 32
verbose = 2

nb_metagenomes = 16
    
def load_data(train_dir, test_dir, model_type):
    
    if model_type == 'cnn' or model_type == 'randomforest' or model_type == 'svm':
        # train
        x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
        y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
        x_train = x_train/(2**8-1)

        # test
        x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
        x_test = x_test/(2**8-1)
        
        return x_train, y_train,x_test, y_test
        
    elif model_type == 'metagenome':
        # train
        x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
        y_train = np.load(os.path.join(train_dir, 'y_train_log10.npy'))
        x_train = x_train/(2**8-1)

        # test
        x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(test_dir, 'y_test_log10.npy'))
        x_test = x_test/(2**8-1)
        test_id = np.load(os.path.join(test_dir, 'test_id.npy'))
        
        return x_train, y_train,x_test, y_test, test_id

    else:
        print("model_type is 'cnn', 'randomforest', 'svm' or 'metagenome' ")
        
def cnn_classification(x_train, y_train, x_test, y_test, save_dir, epoch):
      
    # to_categorical
    y_train = np_utils.to_categorical(y_train)
    true_val = y_test.copy()          
    y_test = np_utils.to_categorical(y_test)
    
    # model train
    conv_base = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(img_rows, img_cols, img_channels)) 
    
    x = conv_base.output
    x = GlobalAveragePooling2D(name='global_average_pooling2D')(x)
    x = Dense(1024, activation = 'relu', name='dense_1')(x)
    predictions = Dense(nb_classes, activation = 'softmax', name='dense_2')(x)

    model = Model(inputs = conv_base.input, outputs = predictions)
        
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adadelta(epsilon=1e-8),
                  metrics=['accuracy'])
                  
    model.fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epoch,
              verbose = verbose,
              shuffle = True,
              validation_data = (x_test, y_test))
    
    ff_out = os.path.join(save_dir, 'model.h5')
    model.save(ff_out)
            
    # predict
    preds = model.predict(x_test)
    pred_val = [np.argmax(pred) for pred in preds]
        
    confusion_matrix = pd.DataFrame(skl_confusion_matrix(true_val, pred_val), index=sample_name,columns=sample_name)
    ff_out = os.path.join(save_dir, 'confusion_matrix.csv')
    confusion_matrix.to_csv(ff_out)
    
def cnn_metagenome(x_train, y_train, x_test, y_test, test_id, save_dir, epoch):
    
    min_value = 4.34783E-05

    def log10(xx):
        numerator = K.log(xx)
        denominator = K.log(K.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    
    def log_activation(xx):
        return log10(K.maximum(xx,min_value))
    
    # model train
    conv_base = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(img_rows, img_cols, img_channels)) 
    
    x = conv_base.output
    x = GlobalAveragePooling2D(name='global_average_pooling2D')(x)
    x = Dense(1024, activation = 'relu', name='dense_1')(x)
    predictions = Dense(nb_metagenomes, activation = 'softmax', name='dense_2')(x)
    
    prediction_log = Activation(log_activation, name='dense_log')(predictions)
    
    model = Model(inputs = conv_base.input, outputs = prediction_log)
    
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adadelta(epsilon=1e-8),
                  metrics = ['mae'])
                  
    model.fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epoch,
              verbose = verbose,
              shuffle = True,
              validation_data = (x_test, y_test))
    
    ff_out = os.path.join(save_dir, 'model.h5')
    model.save(ff_out)
    
    # predict
    ff_out = os.path.join(save_dir, 'predicted.csv')
    predicted =  model.predict(x_test)
    predicted_csv = pd.concat([pd.DataFrame(test_id,columns=['sample']),pd.DataFrame(predicted, columns=metagenome_name)],axis=1)
    predicted_csv.to_csv(ff_out, index=False)

def randomforest_classification(x_train, y_train, x_test, y_test, save_dir):

    s = img_rows * img_cols * img_channels
    
    # train
    nn = x_train.shape[0]
    x_train_tmp = np.zeros((x_train.shape[0],s))
    for j in range(nn):
        x_train_tmp[j,:] = (x_train[j,:].reshape(1,s)[0,])
    
    order = np.arange(nn)
    random.shuffle(order)
    x_train_clf = x_train_tmp[order,:]
    y_train_clf = y_train[order]

    # test
    x_test_clf = np.zeros((x_test.shape[0],s))
    for j in range(x_test.shape[0]):
        x_test_clf[j,:] = (x_test[j,:].reshape(1,s)[0,])
    y_test_clf = y_test 

    clf = RandomForestClassifier(max_depth=10, max_features=10, min_samples_split=5,
                                 n_estimators=500, n_jobs=8)
    clf.fit(x_train_clf, y_train_clf)

    predicted = clf.predict(x_test_clf)
            
    confusion_matrix_tmp = skl_confusion_matrix(y_test_clf, predicted)
    confusion_matrix = pd.DataFrame(confusion_matrix_tmp,index=sample_name, columns=sample_name)
    
    ff_out = os.path.join(save_dir, 'confusion_matrix.csv')
    confusion_matrix.to_csv(ff_out)  

        
def svm_classification(x_train, y_train, x_test, y_test, save_dir):
    
    s = img_rows * img_cols * img_channels
    
    # train
    nn = x_train.shape[0]
    x_train_tmp = np.zeros((x_train.shape[0],s))
    for j in range(nn):
        x_train_tmp[j,:] = (x_train[j,:].reshape(1,s)[0,])
        
    order = np.arange(nn)
    random.shuffle(order)
    x_train_clf = x_train_tmp[order,:]
    y_train_clf = y_train[order]    
    
    # test
    x_test_clf = np.zeros((x_test.shape[0],s))
    for j in range(x_test.shape[0]):
        x_test_clf[j,:] = (x_test[j,:].reshape(1,s)[0,])
    y_test_clf = y_test 
    
    clf = svm.SVC(gamma=0.001, C=1000.)
    clf.fit(x_train_clf, y_train_clf)

    predicted = clf.predict(x_test_clf) 
        
    confusion_matrix_tmp = skl_confusion_matrix(y_test_clf, predicted)
    confusion_matrix = pd.DataFrame(confusion_matrix_tmp,index=sample_name, columns=sample_name)
    
    ff_out = os.path.join(save_dir, 'confusion_matrix.csv')
    confusion_matrix.to_csv(ff_out)  

def merge_confusion_matrix(out_dir):
              
    # merge
    for i in range(5):
        csv = os.path.join(out_dir, 'mouse' + str(i+1), 'confusion_matrix.csv')
        confusion_matrix_tmp = pd.read_csv(csv, index_col=0)
        #confusion_matrix_tmp = pd.read_csv(csv).iloc[:,1:len(SAMPLE_NAME) + 1]
        if i == 0:        
            confusion_matrix = confusion_matrix_tmp
        else:
            confusion_matrix = confusion_matrix +  confusion_matrix_tmp
                
    return confusion_matrix

def plot_confusion_matrix(out_dir, confusion_matrix, prefix, annot=True, cmap='Blues', ratio=True, fmt='.2f'):
    
    total = sum(confusion_matrix.iloc[0,:])
    sum_total =  total * len(sample_name)
    acc_total = sum([confusion_matrix.iat[i,i] for i in range(len(sample_name))])
    
    confusion_matrix_ratio = []
    for i in range(len(sample_name)):
        confusion_matrix_ratio.append([confusion_matrix.iat[i,j]/total for j in range(len(sample_name))])

    confusion_matrix_ratio = np.array(confusion_matrix_ratio)
    
    if ratio:
        plot_matrix = confusion_matrix_ratio
        vmin_val = 0
        vmax_val = 1
    else:
        plot_matrix = confusion_matrix
        vmin_val = None
        vmax_val = None
      
    plt.figure(figsize=(12, 12))

    res = sns.heatmap(plot_matrix, annot=annot, fmt=fmt, vmin=vmin_val, vmax=vmax_val,
                cbar=True, square=True, cmap=cmap, robust=True, cbar_kws={'shrink':0.82},
                xticklabels=sample_name, yticklabels=sample_name)
    
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 12)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 12)
    
    # make frame visible
    for _, spine in res.spines.items():
        spine.set_visible(True)
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.ylim(len(sample_name), 0)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)

    out_fig = os.path.join(out_dir, prefix + '_confusion_matrix.png')
    plt.savefig(out_fig)
    
    plt.close('all')
    
def plot_metagenome(out_dir, metagenome_table):

    # predict
    for i in range(5):
        ff = os.path.join(out_dir, 'mouse' + str(i + 1), 'predicted.csv')
        predicted = pd.read_csv(ff)
        class_names = list(predicted['sample'].unique())
        for j, class_name in enumerate(class_names):
            predicted_median_tmp = predicted.loc[predicted['sample'] == class_name,:].median(axis=0)
            if j == 0:
                predicted_median = predicted_median_tmp
            else:
                predicted_median = pd.concat([predicted_median, predicted_median_tmp], axis=1)
        
        predicted_median.index = metagenome_table.index
        predicted_median.columns = class_names
        if i == 0:
            predicted_median_all = predicted_median
        else:
            predicted_median_all = pd.concat([predicted_median_all, predicted_median], axis=1)
        
    predicted_median_all = predicted_median_all[metagenome_table.columns.values]
    
    ff_out = os.path.join(out_dir, 'metagenome_median.csv')
    predicted_median_all.to_csv(ff_out)

    # plot all
    plt.figure(figsize=(6,6))
    for i in reversed(range(16)):
        plt.plot(metagenome_table.iloc[i,:],predicted_median_all.iloc[i,:],'o', color=plt.cm.tab20.colors[i])
        plt.xlim([-4.8,0])
        plt.ylim([-4.8,0])
        plt.xlabel('Observed',fontsize=18)
        plt.ylabel('Predicted',fontsize=18)
    plt.plot([-4.8,0],[-4.8,0],color = 'gray', linestyle='dashed',linewidth=0.5)
    
    out_fig = os.path.join(out_dir, 'metagenome_median.png')
    plt.savefig(out_fig)
    
    plt.close('all')

    # plot each
    fig, axes = plt.subplots(4, 4, figsize=(20,20))

    for i, ax in enumerate(axes.flat):
        ob = metagenome_table.iloc[i,:]
        pr = predicted_median_all.iloc[i,:]
    
        ax.plot(ob,pr,'o', color=plt.cm.tab20.colors[i])
    
        ob_pr = pd.concat([ob,pr])
        min_val = ob_pr.min()
        max_val = ob_pr.max()
        alpha = (max_val - min_val) / 10
        min_max_val = [min_val - alpha, max_val + alpha]
    
        ax.set_xlim(min_max_val)
        ax.set_ylim(min_max_val)
        ax.set_title(metagenome_table.index.values[i], fontsize=16)
    
        ax.plot(min_max_val,min_max_val,color = 'gray', linestyle='dashed',linewidth=0.5)
        
    fig.text(0.5, 0.08, 'Observed', va='center', ha='center', fontsize=25)
    fig.text(0.08, 0.5, 'Predicted', va='center', ha='center', rotation='vertical', fontsize=25)
        
    out_fig = os.path.join(out_dir, 'metagenome_median_each.png')
    plt.savefig(out_fig)
        
    plt.close('all')
