import setGPU
import os

import keras
import keras.backend as K
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv1D, Flatten, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import data
import argparse
from importlib import import_module
import time

import logging

import data

#logging.basicConfig(level=logging.DEBUG)
if __name__ == '__main__':
    # location of data                                                                                                            
    train_val_fname = '/storage/user/jduarte/DNNTuples/train/train_file_*.h5'
    test_fname = '/storage/user/jduarte/DNNTuples/test/test_file_*.h5'
    example_fname = '/storage/user/jduarte/DNNTuples/train/train_file_900.h5'
                                 
    parser = argparse.ArgumentParser(description="train pfcands",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-epochs', type=int, default=500,
                        help='max num of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='the optimizer type')
    parser.add_argument('--model-prefix', type=str, default='models/',
                        help='model prefix')
    parser.add_argument('--disp-batches', type=int, default=20,
                        help='show progress for every n batches')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='run prediction instead of training')

    data.add_data_args(parser)
    parser.set_defaults(
        # config
        model_prefix='models/',
        disp_batches=20,
        # data
        data_config='data_ak8_pfcand_reduced_cloud',
        data_train=train_val_fname,
        train_val_split=0.8,
        data_test=test_fname,
        data_example=example_fname,
        data_names=None,
        num_examples=-1,
        # train
        batch_size=1024,
        num_epochs=200,
        optimizer='adam',
        lr=1e-3,
    )
    args = parser.parse_args()
    dd = import_module(args.data_config)
    args.data_names = ','.join(dd.train_groups)

    (train, val) = dd.load_data(args)

    n_train_val, n_test = dd.nb_samples([args.data_train, args.data_test])
    n_train = int(n_train_val * args.train_val_split)
    n_val = int(n_train_val * (1 - args.train_val_split))


    if args.num_examples < 0:
        args.num_examples = n_train

    nfeatures = 27
    ncands = 100
    nlabels = 7

    # define dense keras model
    inputs = Input(shape=(ncands,nfeatures,), name = 'input')  
    x = BatchNormalization(name='bn_1')(inputs)
    x = Conv1D(64, 1, name = 'conv1d_1', activation='relu')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Conv1D(32, 1, name = 'conv1d_2', activation='relu')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Conv1D(32, 1, name = 'conv1d_3', activation='relu')(x)
    x = BatchNormalization(name='bn_4')(x)
    x = Lambda( lambda x: K.mean(x, axis=-2), input_shape=(ncands,32)) (x)
    x = Dense(100, name='dense_1', activation='relu')(x)
    x = Dense(100, name='dense_2', activation='relu')(x)
    outputs = Dense(nlabels, name = 'output', activation='softmax')(x)
    keras_model = Model(inputs=[inputs], outputs=[outputs])
    keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(keras_model.summary())


    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]
    
    # fit keras model
    keras_model.fit_generator(train, #train.inf_generate_data(), 
                              steps_per_epoch=n_train//args.batch_size, 
                              epochs=args.num_epochs, 
                              validation_data=val, #val.inf_generate_data(),
                              validation_steps=n_val//args.batch_size,
                              shuffle=False,
                              callbacks = callbacks)
    


    # reload best weights
    keras_model.load_weights('keras_model_best.h5')

    # run model inference on test data set
    predict_test = keras_model.predict(X_test)

    # create ROC curve
    fpr = []
    tpr = []
    threshold = []
    acc = []
    for i in range(y_test.shape[1]):
        f, t, th = roc_curve(y_test[:,i],predict_test[:,i])
        fpr.append(f)
        tpr.append(t)
        threshold.append(th)
    
    acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(predict_test,axis=1))
    
    print('accuracy', acc*100)

    plt.figure()
    for i in range(y_test.shape[1]):
        plt.plot(tpr[i], fpr[i], lw=2.5, label="{} AUC = {:.1f}%".format(label[i],auc(fpr[i],tpr[i])*100))
    plt.xlabel(r'True positive rate')
    plt.ylabel(r'False positive rate')
    plt.semilogy()
    plt.ylim(0.001,1)
    plt.xlim(0,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('ROC.png')
    plt.savefig('ROC.pdf')

