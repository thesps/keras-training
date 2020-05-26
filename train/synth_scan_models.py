import hls4ml
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
from joblib import Parallel, delayed

hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
btnn_passes = ['eliminate_linear_activation', 'merge_batch_norm_quantized_tanh', 'quantize_dense_output',
               'fuse_dense_batch_norm', 'fuse_biasadd', 'qkeras_factorize_alpha', 'fuse_consecutive_batch_normalization']

co = {}; _add_supported_quantized_objects(co)

def build(i):
    model = tf.keras.models.load_model('scan_models/model_{}/KERAS_check_best_model.h5'.format(i), custom_objects=co)
    hls4ml_cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls4ml_cfg['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,9>'
    hls4ml_cfg['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
    if i < 3:
        #hls4ml_cfg['Model'] = {}
        #hls4ml_cfg['Model']['Precision'] = 'ap_fixed<16,6>'
        #hls4ml_cfg['Model']['ReuseFactor'] = 1
        #hls4ml_cfg['Model']['Optimizers'] = btnn_passes
        hls4ml_cfg['Optimizers'] = btnn_passes
    hls_model = hls4ml.converters.convert_from_keras_model(model, output_dir='scan_models/model_{}/hls4ml_prj'.format(i),
                                                           fpga_part='xcvu9p-flgb2104-2L-e', hls_config=hls4ml_cfg)
    hls_model.compile()
    hls_model.build(csim=False, export=False, cosim=False, synth=True, vsynth=True)

if __name__ == '__main__':
    Parallel(n_jobs=4, backend='multiprocessing')(delayed(build)(i) for i in range(1,17))
    #for i in range(3, 17):
    #    build(i)
