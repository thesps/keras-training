from hls4ml.model.profiling import weights_keras, activations_keras 
from hls4ml.model.hls_model import Input, Dense, Activation
import numpy as np
from sklearn.metrics import accuracy_score

def inc_dir(hlsconfig):
    directory = hlsconfig['OutputDir']
    directory = directory.split('_')
    n = int(directory[-1])
    directory = '_'.join(directory[:-1]) + '_' + str(n+1)
    hlsconfig['OutputDir'] = directory
    return hlsconfig

def accuracy(y_test, y_pred):
    return accuracy_score(y_test, np.argmax(y_pred, axis=1))

def new_predict(hls_model, hlsconfig, X):
    '''Create a new HLSModel, compile it, and return the predictions.'''
    meta_cfg = hls_model.config
    hlsmodel = hls4ml.converters.keras_to_hls(hlsconfig)
    hlsmodel.compile()
    return hlsmodel.predict(X)


def iter_precision_layer(keras_model, y_test, acc_keras, eps, hls_model, hlsconfig, layername, var, I_init, W_init):
    I, W = I_init, W_init
    hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var] = 'ap_fixed<{},{}>'.format(W, I)
    hlsconfig = inc_dir(hlsconfig)
    acc_hls = accuracy(y_test, new_predict(hls_model, hlsconfig, X))
    print(hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var], acc_hls, abs(acc_keras - acc_hls) / acc_keras)
    while(abs(acc_keras - acc_hls) / acc_keras < eps):
        I -= 1
        hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var] = 'ap_fixed<{},{}>'.format(W, I)
        hlsconfig = inc_dir(hlsconfig)
        acc_hls = accuracy(y_test, new_predict(hls_model, hlsconfig, X))
        print(hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var], acc_hls, abs(acc_keras - acc_hls) / acc_keras)
    if abs(acc_keras - acc_hls) / acc_keras >=  eps:
        I += 1
    hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var] = 'ap_fixed<{},{}>'.format(W, I)
    hlsconfig = inc_dir(hlsconfig)
    acc_hls = accuracy(y_test, new_predict(hls_model, hlsconfig, X))
    print(hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var], acc_hls, abs(acc_keras - acc_hls) / acc_keras)
    while(abs(acc_keras - acc_hls) / acc_keras < eps):
        W -= 1
        hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var] = 'ap_fixed<{},{}>'.format(W, I)
        hlsconfig = inc_dir(hlsconfig)
        acc_hls = accuracy(y_test, new_predict(hls_model, hlsconfig, X))
        print(hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var], acc_hls, abs(acc_keras - acc_hls) / acc_keras)
    if abs(acc_keras - acc_hls) / acc_keras >=  eps:
        W += 1
    hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var] = 'ap_fixed<{},{}>'.format(W, I)
    hlsconfig = inc_dir(hlsconfig)
    acc_hls = accuracy(y_test, new_predict(hls_model, hlsconfig, X))
    print(hlsconfig['HLSConfig']['LayerName'][layername]['Precision'][var], acc_hls, abs(acc_keras - acc_hls) / acc_keras)
    return hlsconfig

def iter_precision_model(keras_model, hls_model, X, y, eps, startw):
    ''' Iteratively reduce the fixed-point precision of a model.
        Currently only Dense layers are affected.
        Iterates in the order: layer, [weight, bias, result], [integer bits, width bits]
        Test data must be provided, as well as an 'epsilon' acceptable accuracy loss and starting width.
        Initial values for the integer bits are obtained from model profiling.

        Arguments:
        keras_model : the Keras model to optimize
        hls_model :   the HLS Model to iterate. This can just have a basic configuration.
        X :           test features
        y :           test truth
        eps :         epsilon parameter. When HLS accuracy differs from the Keras accuracy by more than epsilon,
                      iteration moves onto the next parameter. 
                      eps can be a single float, or an array of floats with the same length as len(keras_model.layers).
                      In this case the i'th epsilon entry is used for the i'th layer.
        startw :      integer: the width to initialize all variables to

        Returns: the optimized HLSConfig
    
    '''
    X_test, y_test = X, y
    y_pred = keras_model.predict(X_test)
    y_hls = hls_model.predict(X_test)
    acc_keras = accuracy(y_test, y_pred)
    acc_hls = accuracy(y_test, y_hls)
    print("Keras accuracy:     {}".format(accuracy(y_test, y_pred)))
    print("HLS   accuracy: {}".format(accuracy(y_test, y_hls)))
    hlsconfig = hls_model.config.config
    hlsconfig['HLSConfig']['LayerName'] = {}
    layers = list(hls_model.get_layers())
    # init config
    w, a = weights_keras(model, fmt='summary'), activations_keras(model, X, fmt='summary')
    w_prof, a_prof = {}, {}
    for wi in w:
        w_prof[wi['weight']] = wi
    for ai in a:
        a_prof[ai['weight']] = ai

    # Do the iteration
    for il, layer in enumerate(layers):
        if isinstance(layer, Dense):
            if isinstance(eps, list):
                epsi = eps[il]
            else:
                epsi = eps
            I_w, W_w = int(np.ceil(np.log2(w_prof[layer.name + '/0']['whishi']))) + 3, startw
            I_b, W_b = int(np.ceil(np.log2(w_prof[layer.name + '/1']['whishi']))) + 3, startw
            I_a, W_a = int(np.ceil(np.log2(a_prof[layer.name]['whishi']))) + 3, startw
            
            # do w
            hlsconfig['HLSConfig']['LayerName'][layer.name] = {}
            hlsconfig['HLSConfig']['LayerName'][layer.name]['Precision'] = {}
            start = {}
            for quantizable in ['weight', 'bias', 'result']:
                start[quantizable] = {}
            start['weight']['I'], start['weight']['W'] = I_w, W_w
            start['bias']['I'], start['bias']['W'] = I_b, W_b
            start['result']['I'], start['result']['W'] = I_a, W_a

            for q in ['weight', 'bias', 'result']:
                print(q)
                hlsconfig = iter_precision_layer(keras_model, y_test, acc_keras, epsi, hls_model, hlsconfig, layer.name, q, start[q]['I'], start[q]['W'])
    return hlsconfig

if __name__ == "__main__":
    import tensorflow as tf; import hls4ml; import matplotlib.pyplot as plt; import yaml;
    from plot_scan import model_performance_summary, extract_data_baseline, get_Xyl
    Xyl = get_Xyl(); X, y, l = Xyl; X = X[:5000]; y = y[:5000]
    model = tf.keras.models.load_model('baseline/KERAS_check_best_model.h5')
    hls4ml_cfg = open('baseline/hls4ml_cfg.yml')
    hls4ml_cfg = yaml.load(hls4ml_cfg, Loader=yaml.SafeLoader)
    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           output_dir='baseline/hls4ml_prj_0',
                                                           fpga_part='xcvu9p-flgb2104-2L-e',
                                                           hls_config=hls4ml_cfg)
    hls_model.compile()
    #eps = [0, 0.01, 0, 0.015, 0, 0.02, 0, 0.02, 0]
    #eps = [0, 0.006, 0, 0.008, 0, 0.01, 0, 0.01, 0]
    eps = [0, 0.004, 0, 0.006, 0, 0.008, 0, 0.008, 0]
    hlsconfig = iter_precision_model(model, hls_model, X, y, eps, 18)
    
    def write_dict(d, f, depth=0):
        if isinstance(d, dict):
            for key in d.keys():
                f.write(''.join(['  ' for i in range(depth)]))
                f.write(key + ':\n')
                write_dict(d[key], f, depth=depth+1)
        else:
            f.write(''.join(['  ' for i in range(depth+1)]))
            f.write(str(d) + '\n')

    #f = open('baseline/hls4ml_cfg_eps' + str(eps) + '.yml', 'w')
    #write_dict(hlsconfig['HLSConfig'], f)
    #f.close()
