import tensorflow as tf
from train import get_features
import yaml
from qkeras import utils
import hls4ml

co = {}
utils._add_supported_quantized_objects(co)
model = tf.keras.models.load_model('QKeras_model_4b0.h5', custom_objects=co)
# Try a wide data type for debugging
#qconfig = hls4ml.utils.config_from_keras_model(model, granularity='name')
qconfig = hls4ml.utils.config_from_keras_model(model, granularity='model')
qconfig['Model']['Precision'] : 'ap_fixed<32,16>'
print(qconfig)
hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=qconfig)

def get_opts():
    class dummy:
        def __init__(self):
            self.x = 0
    opts = dummy()
    opts.tree = 't_allpar_new'
    opts.inputFile = '../../data/jet-tagging/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z'
    opts.outputDir = './test'
    return opts

opts = get_opts()

cfg = open('train_config_threelayer_ternary.yml')
cfg = yaml.load(cfg, Loader=yaml.SafeLoader)
X_train_val, X_test, y_train_val, y_test, labels = get_features(opts, cfg)
X = X_test[:10]

hls_model.compile()
yh = hls_model.predict(X)
yq = model.predict(X)
