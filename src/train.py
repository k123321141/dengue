from tqdm import tqdm
import numpy as np
import os
import keras
import pickle
import tensorflow as tf
import json
from keras.layers import Input, Dense, BatchNormalization, Dropout, Add, Concatenate
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import KFold
from keras.backend.tensorflow_backend import set_session
from sklearn.linear_model import LogisticRegression
import argparse


def init_sess():
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def init_model(input_dim, dm, drop, residual_type, depth, act_fn):
    inputs = Input(shape=(input_dim,))
    x = Dense(dm, activation='linear')(inputs)
    if residual_type == 'DenseNet_1' or residual_type == 'DenseNet_2':
        des_buf = []
    for i in range(depth):
        if residual_type == 'ResNet':
            x1 = Dense(2*dm, activation=act_fn)(x)
            x2 = Dense(dm, activation='linear')(x1)
            x = Add()([x, x2])
            x = BatchNormalization()(x)
            x = Dropout(drop, name='output_at_depth_%d' % (i+1))(x)
        elif residual_type == 'None':
            x1 = Dense(2*dm, activation=act_fn)(x)
            x2 = Dense(dm, activation='linear')(x1)
            x = x2
            x = BatchNormalization()(x)
            x = Dropout(drop, name='output_at_depth_%d' % (i+1))(x)
        elif residual_type == 'DenseNet_1':
            if i < 2:
                x = x
            else:
                x = Add()(des_buf)
            x = Dense(dm, activation=act_fn)(x)
            x = BatchNormalization()(x)
            x = Dropout(drop, name='output_at_depth_%d' % (i+1))(x)
            des_buf.append(x)
        elif residual_type == 'DenseNet_2':
            if i < 2:
                x = x
            else:
                x = Concatenate(axis=-1)(des_buf)
            x = Dense(dm, activation=act_fn)(x)
            x = BatchNormalization()(x)
            x = Dropout(drop, name='output_at_depth_%d' % (i+1))(x)
            des_buf.append(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=y)
    return model


def train_phase(X, Y, config):
    model = init_model(**config)
    batch = 2048
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(
            X, Y, batch_size=batch, epochs=6000,
            verbose=0,
            class_weight={0: 1, 1: 5},
    )  # starts training
    return model


def train_phase_logistic_regression(X, Y, config):
    Y = Y.flatten()
    clf = LogisticRegression(
            solver='lbfgs',
            penalty='l2',
            class_weight={0: 1, 1: 1},
            max_iter=100
    )
    m = clf.fit(X, Y)
    m.predict = lambda X: m.predict_proba(X)[:, 1]
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dengue paper training code.")
    parser.add_argument('model', type=str, choices=['NN', 'LR'],
                        help='Specify the trained model path.')
    parser.add_argument('feature_count', type=int, choices=[6, 11, 18],
                        help='Specify the feature count.')
    parser.add_argument('--seed', type=int, default=9487,
                        help='Specify random seed.')
    parser.add_argument('--data', type=str, default='../data/data.npz',
                        help='Specify input path.')
    parser.add_argument('-o', '--output', type=str, default='./outputs',
                        help='Specify the output path.')
    # NN model setting
    parser.add_argument('--dm', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--residual_type', type=str, default='DenseNet_1')
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--act_fn', type=str, default='relu')

    args = parser.parse_args()
    data = np.load(args.data)
    # apply input feature filter
    F = data['feature_names'].tolist()
    if args.feature_count == 6:
        iid = [F.index(f) for f in ['Temp', 'age', 'exam_WBC', 'exam_Plt', 'exam_Hb', 'sex']]
    elif args.feature_count == 11:
        iid = [F.index(f) for f in ['Temp', 'age', 'SBP', 'DBP', 'Breath', 'Pulse', 'GCS_Total', 'exam_WBC', 'exam_Plt', 'exam_Hb', 'sex']]
    elif args.feature_count == 18:
        iid = [F.index(f) for f in ['Temp', 'age', 'SBP', 'DBP', 'Breath', 'Pulse', 'GCS_Total', 'exam_WBC', 'exam_Plt', 'exam_Hb', 'sex', 'Cancer', 'Hypertension', 'Heart Disease', 'CVA', 'CKD', 'Severe Liver Disease', 'DM']]
    X = data['x'][:, iid]
    v_mask = data['missing_mask']
    Y = data['y']
    X = X[v_mask, :]
    Y = Y[v_mask, :]
    # parse default model hyper-parameter
    model_type = args.model
    if model_type == 'NN':
        config = {
            'input_dim': args.feature_count,
            'dm': args.dm,
            'drop': args.dropout,
            'residual_type': args.residual_type,
            'depth': args.depth,
            'act_fn': args.act_fn
        }
    elif model_type == 'LR':
        config = {
            'solver': 'lbfgs',
            'penalty': 'l2',
            'class_weight': {0: 1, 1: 1},
            'max_iter': 100
        }

    print 'input shape : %s' % str(X.shape)
    print 'positive sample count : %s' % sum(Y)
    # training setting
    np.random.seed(args.seed)
    fold_num = 10
    min_recall = 0.9
    N = len(X)
    iters = 20
    outputs = {}
    with tqdm(total=fold_num*iters) as pbar:
        for i in range(iters):
            out_buf = []
            test_idx = []
            kf = KFold(n_splits=fold_num, shuffle=True)
            for it, (train_index, test_index) in enumerate(kf.split(Y.flatten())):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                if model_type == 'LR':
                    model = train_phase_logistic_regression(X_train, y_train, config)
                elif model_type == 'NN':
                    init_sess()
                    model = train_phase(X_train, y_train, config)
                else:
                    raise NotImplementedError
                test_idx.append(test_index)
                out_buf.append(model.predict(X_test))
                pbar.update(1)
            test_idx = np.concatenate(test_idx, axis=0)
            inverted_idx = np.argsort(test_idx)
            out = np.concatenate(out_buf, axis=0).flatten()[inverted_idx]
            outputs[i] = out
    # output
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    output_path = os.path.join(args.output, '%s-%s.output' % (model_type, args.feature_count))
    with open(output_path, 'wb') as fout:
        pickle.dump(outputs, fout)
    output_path = os.path.join(args.output, '%s-%s.config' % (model_type, args.feature_count))
    with open(output_path, 'wb') as fout:
        json.dump(config, fout)
    print 'Writed to %s' % output_path
