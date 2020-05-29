from train import get_features, default_options_and_config
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from callbacks import all_callbacks
from qkeras.utils import model_quantize
import sys
sys.path.insert(0, '../models')
import models
from joblib import Parallel, delayed


qtemplate = {'default'  : {'kernel_quantizer' : None, 'bias_quantizer' : None},
             'QDense' : {'kernel_quantizer' : None, 'bias_quantizer' : None},
             'QActivation' : {'relu' : None}}

def build_model(i, X, y, opts, cfg):
    if i == 1:
        quantizer = 'binary(alpha=1)'
        #act_quantizer = 'binary_tanh()'
    elif i == 2:
        quantizer = 'ternary(alpha=1)'
        #act_quantizer = 'quantized_tanh(2,2)'
    else:
        quantizer = 'quantized_bits({},0,alpha=1)'.format(i)
        #act_quantizer = 'quantized_relu({})'.format(i)

    act_quantizer='quantized_relu(4)'
    for type_key in ['default', 'QDense']:
        for q_key in ['kernel_quantizer', 'bias_quantizer']:
            qtemplate[type_key][q_key] = quantizer
    qtemplate['QActivation']['relu'] = act_quantizer

    opts.outputDir = "scan_models_relu4/model_{}".format(i)

    model = getattr(models, cfg['KerasModel'])
    model = model(Input(shape=X.shape[1:]), y.shape[1], l1Reg=cfg['L1Reg'])
    model = model_quantize(model, qtemplate, i)
    return model

def train_model(model, X, y, opts, cfg):
    startlearningrate=0.0001
    adam = Adam(lr=startlearningrate)
    model.compile(optimizer=adam, loss=[cfg['KerasLoss']], metrics=['accuracy'])


    callbacks=all_callbacks(stop_patience=1000,
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001,
                            lr_cooldown=2,
                            lr_minimum=0.0000001,
                            outputDir=opts.outputDir)

    model.fit(X, y, batch_size = 1024, epochs = 100,
                    validation_split = 0.25, shuffle = True, callbacks = callbacks.callbacks)
    return model

def build_train_model(i, X, y, opts, cfg):
    model = build_model(i, X, y, opts, cfg)
    model = train_model(model, X, y, opts, cfg)

if __name__ == '__main__':
    opts, cfg = default_options_and_config()
    X_train_val, X_test, y_train_val, y_test, labels  = get_features(opts, cfg)
    Parallel(n_jobs=4)(delayed(build_train_model)(i, X_train_val, y_train_val, opts, cfg) for i in range(1, 17))
    
