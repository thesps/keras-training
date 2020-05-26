import hls4ml
from optparse import OptionParser
import pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
from train import get_features, default_options_and_config
import yaml

hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

avail = {'dsp' : 6840, 'lut' : 1182240, 'ff' : 2364480}

def get_Xyl():
    '''Load the default dataset and return (X_test, y_test, labels) tuple'''
    opts, cfg = default_options_and_config()
    X_train_val, X_test, y_train_val, y_test, labels  = get_features(opts, cfg)
    ohe = OneHotEncoder().fit(np.arange(0,5,1).reshape(-1,1))
    y_test = ohe.inverse_transform(y_test)
    Xyl = (X_test, y_test, labels)
    return Xyl

def model_performance_summary(model, hls_model, Xyl, report=False):
    '''Get the performance summary (model and synthesis) for a model and corresponding hls_model
       args:
        model: a (Q)Keras model
        hls_model: the corresponding hls4ml HLSModel object
        Xyl: a tuple of (X_test, y_test, labels) e.g. returned by get_Xyl
        report: boolean, whether or not to load the HLS synthesis reports
    '''
    X_test, y_test, labels = Xyl
    y_predict = model.predict(X_test)
    y_predict_hls4ml = hls_model.predict(X_test)

    data = {}
    one = np.ones_like(y_test.shape[0])
    zero = np.zeros_like(y_test.shape[0])
    for j in range(len(labels)):
        label = labels[j].replace('j_','')
        y_class = (y_test == j).flatten().reshape(y_test.shape[0])
        y_binary = np.where(y_class, one, zero)
        fpr, tpr, thresh = roc_curve(y_binary, y_predict[:,j])
        data['auc_qkeras_' + label] = auc(fpr, tpr)

        fpr, tpr, thresh = roc_curve(y_binary, y_predict_hls4ml[:,j])
        data['auc_hls4ml_' + label] = auc(fpr, tpr)

    data['accuracy_qkeras'] = accuracy_score(y_test, np.argmax(y_predict,axis=1).reshape(-1,1))
    data['accuracy_hls4ml'] = accuracy_score(y_test, np.argmax(y_predict_hls4ml,axis=1).reshape(-1,1))

    if report:
        # Get the resources from the logic synthesis report
        report = open('{}/vivado_synth.rpt'.format(hls_model.config.get_output_dir()))
        lines = np.array(report.readlines())
        data['lut'] = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2]) 
        data['ff'] = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
        data['bram'] = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2]) 
        data['dsp'] = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
        report.close()

        # Get the latency from the Vivado HLS report
        report = open('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(hls_model.config.get_output_dir()))
        lines = np.array(report.readlines())
        lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
        data['latency_clks'] = int(lat_line.split('|')[2])
        data['latency_ns'] = float(lat_line.split('|')[4].replace('ns',''))
    return data

def extract_data_scan(scan_dir, Xyl):
    co = {}
    _add_supported_quantized_objects(co)

    data = {'w':[], 'accuracy_qkeras':[], 'accuracy_hls4ml' : [], 'dsp':[], 'lut':[], 'ff':[],
            'bram':[], 'latency_clks':[], 'latency_ns':[]}
    for label in ['q','g','t','w','z']: data['auc_qkeras_' + label] = []
    for label in ['q','g','t','w','z']: data['auc_hls4ml_' + label] = []

    for i in range(3,17):
        model = tf.keras.models.load_model('{}/model_{}/KERAS_check_best_model.h5'.format(scan_dir, i), custom_objects=co)
        hls4ml_cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
        hls4ml_cfg['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,9>'
        hls4ml_cfg['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
        hls_model = hls4ml.converters.convert_from_keras_model(model, output_dir='{}/model_{}/hls4ml_prj_eval'.format(scan_dir, i),
                                                       fpga_part='xcvu9p-flgb2104-2L-e', hls_config=hls4ml_cfg)
        hls_model.compile()
        datai = model_performance_summary(model, hls_model, Xyl, reports=True)
        for key in datai.keys():
            data[key].append(datai[key])

    data = pandas.DataFrame(data)
    return data

def extract_data_baseline_unpruned(Xyl):
    model = tf.keras.models.load_model('baseline/KERAS_check_best_model.h5')
    hls4ml_cfg = hls4ml.utils.config_from_keras_model(model, granularity='Model')
    data = {'w':[], 'accuracy_qkeras':[], 'accuracy_hls4ml' : [], 'dsp':[], 'lut':[], 'ff':[],
            'bram':[], 'latency_clks':[], 'latency_ns':[]}
    for label in ['q','g','t','w','z']: data['auc_qkeras_' + label] = []
    for label in ['q','g','t','w','z']: data['auc_hls4ml_' + label] = []
    for i in range(3,17):
        hls4ml_cfg['Model']['Precision'] = 'ap_fixed<{},6>'.format(i)
        hls_model = hls4ml.converters.convert_from_keras_model(model, output_dir='baseline/hls4ml_prj_{}'.format(i),
                    fpga_part='xcvu9p-flgb2104-2L-e', hls_config=hls4ml_cfg)
        hls_model.compile()
        datai = model_performance_summary(model, hls_model, Xyl, report=False)
        for key in datai.keys():
            data[key].append(datai[key])

    return data

def extract_data_baseline(Xyl):
    model = tf.keras.models.load_model('baseline/KERAS_pruned_model.h5')
    hls4ml_cfg = open('baseline/hls4ml_cfg.yml')
    hls4ml_cfg = yaml.load(hls4ml_cfg, Loader=yaml.SafeLoader)
    hls_model = hls4ml.converters.convert_from_keras_model(model, output_dir='baseline/hls4ml_prj_eval',
                fpga_part='xcvu9p-flgb2104-2L-e', hls_config=hls4ml_cfg)
    hls_model.compile()
    data = model_performance_summary(model, hls_model, Xyl, report=False)
    return data


def make_plots(data):
    f0 = plt.figure()
    plt.plot(data['w'], data['dsp'] * 10, label=r'DSP \times 10')
    plt.plot(data['w'], data['lut'], label=r'LUT')
    plt.plot(data['w'], data['ff'], label=r'FF')
    plt.legend()
    plt.xlabel('Bitwidth')
    plt.ylabel('Resource Usage')

    f1 = plt.figure()
    plt.plot(data['w'], data['dsp'] / avail['dsp'], label=r'DSP')
    plt.plot(data['w'], data['lut'] / avail['lut'], label=r'LUT')
    plt.plot(data['w'], data['ff'] / avail['ff'], label=r'FF')
    plt.legend()
    plt.xlabel('Bitwidth')
    plt.ylabel('Resource Usage (% VU9P)')

    f2 = plt.figure()
    plt.plot(data['w'], data['accuracy_qkeras'], label='QKeras')
    plt.plot(data['w'], data['accuracy_hls4ml'], label='hls4ml')
    plt.xlabel('Bitwidth')
    plt.ylabel('Accuracy')
    return f0, f1, f2

def do_plots(percent_resources=False, log=False):
    import pandas; import plt4hls4ml; plt4hls4ml.init()
    data = pandas.read_csv('scan_models/summary.csv')
    data0 = pandas.read_csv('baseline/summary_eps1_rising_2.csv')
    data1 = pandas.read_csv('baseline_junior/summary_optimized.csv')
    baseline = pandas.read_csv('baseline/summary_unpruned_scan.csv')
    baseline0 = pandas.read_csv('baseline/summary_full_14_6.csv')
    baseline1 = pandas.read_csv('baseline/summary_pruned_14_6.csv')

    symbols = ['P', 'o', 'X', 'D']
    d = [baseline0, baseline1, data0, data1]
    names = ['B', 'BP', 'BH', 'QO']

    norm_lut = 100. / avail['lut'] if percent_resources else 1
    norm_ff = 100. / avail['ff'] if percent_resources else 1
    norm_dsp = 100. / avail['dsp'] if percent_resources else 10
    width = [3,1]
    f = plt.figure(figsize=(3,3), constrained_layout=True)
    gs = f.add_gridspec(ncols=2, nrows=1, width_ratios=width)
    ax0 = f.add_subplot(gs[0,0])
    plt.plot(data['w'], data['lut'] * norm_lut, label=r'LUT')
    plt.plot(data['w'], data['ff'] * norm_ff, label=r'FF')
    plt.plot(data['w'], data['dsp'] * norm_dsp, label=r'DSP' if percent_resources else r'DSP $\times 10$' )
    plt.gca().set_prop_cycle(None)
    #plt.plot(data[data['w'] == 6]['w'], data[data['w'] == 6]['lut'] * norm_lut, 'x')
    #plt.plot(data[data['w'] == 6]['w'], data[data['w'] == 6]['ff'] * norm_ff, 'x')
    #plt.plot(data[data['w'] == 6]['w'], data[data['w'] == 6]['dsp'] * norm_dsp, 'x')
    #plt.gca().ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    plt.legend()
    plt.xlabel('Bitwidth')
    plt.ylabel('Resource Usage (%)')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))

    ax1 = f.add_subplot(gs[0,1], sharey=ax0)
    x = 0

    plt.gca().set_prop_cycle(None)
    for i, di in enumerate(d):
        s = symbols[i]
        plt.plot(x, di['lut'] * norm_lut, s)
        plt.plot(x, di['ff'] * norm_ff, s)
        plt.plot(x, di['dsp'] * norm_dsp, s)
        x += 1
        plt.gca().set_prop_cycle(None)
    ax1.set_xlim((-0.5, len(d) - 0.5))
    ax1.get_yaxis().set_visible(False)
    plt.xticks(list(range(len(names))), names, fontsize=6)
    if log:
        plt.semilogy()

    plt.savefig('resources.pdf')

    f1 = plt.figure(figsize=(3,3), constrained_layout=True)
    gs = f1.add_gridspec(ncols=2, nrows=1, width_ratios=width)
    ax0 = f1.add_subplot(gs[0,0])
    plt.plot(data['w'], data['accuracy_qkeras'] / baseline['accuracy_qkeras'].iloc[0], label='QKeras CPU')
    plt.plot(data['w'], data['accuracy_hls4ml'] / baseline['accuracy_qkeras'].iloc[0], label='QKeras FPGA')
    plt.plot(baseline['w'], baseline['accuracy_hls4ml'] / baseline['accuracy_qkeras'], '--', label='Post-train quant.')
    plt.gca().set_prop_cycle(None)
    #plt.plot(data[data['w'] == 6]['w'], data[data['w'] == 6]['accuracy_qkeras'] / baseline['accuracy_qkeras'].iloc[0], 'x')
    #plt.plot(data[data['w'] == 6]['w'], data[data['w'] == 6]['accuracy_hls4ml'] / baseline['accuracy_qkeras'].iloc[0], 'x')
    plt.xlabel('Bitwidth')
    plt.ylabel('Ratio Model Accuracy / Baseline Accuracy')
    plt.ylim((0.9, 1.05))
    plt.legend(loc='upper left')

    ax1 = f1.add_subplot(gs[0,1], sharey=ax0)
    plt.gca().set_prop_cycle(None)
    x = 0
    c1 = plt4hls4ml.colors[1]
    c2 = plt4hls4ml.colors[2]
    colors = [c2, c2, c2, c1]
    for i, di in enumerate(d):
        s = symbols[i]
        plt.plot(x, di['accuracy_hls4ml'] / baseline['accuracy_qkeras'].iloc[0], s, color=colors[i])
        #plt.gca().set_prop_cycle(None)
        x += 1
    ax1.set_xlim((-0.5, len(d) - 0.5))
    ax1.get_yaxis().set_visible(False)
    plt.xticks(list(range(len(names))), names, fontsize=6)

    plt.savefig('accuracy.pdf')


    f1 = plt.figure(figsize=(3,3), constrained_layout=True)
    gs = f1.add_gridspec(ncols=2, nrows=1, width_ratios=width)
    ax0 = f1.add_subplot(gs[0,0])
    for label in ['g', 'q', 'w', 'z', 't']:
        y = data['auc_hls4ml_{}'.format(label)] / baseline['auc_qkeras_{}'.format(label)]
        plt.plot(data['w'], y, label=label)
    plt.legend()

    plt.gca().set_prop_cycle(None)
    for label in ['g', 'q', 'w', 'z', 't']:
        y = baseline['auc_hls4ml_{}'.format(label)] / baseline['auc_qkeras_{}'.format(label)]
        plt.plot(baseline['w'], y, '--')

    plt.xlabel('Bitwidth')
    plt.ylabel('Ratio AUC QKeras+hls4ml / Keras float')

    ax1 = f1.add_subplot(gs[0,1], sharey=ax0)
    plt.gca().set_prop_cycle(None)
    for label in ['g', 'q', 'w', 'z', 't']:
        y = data0['auc_hls4ml_{}'.format(label)] / baseline['auc_qkeras_{}'.format(label)].iloc[0]
        plt.plot(0, y, 'P')
    plt.gca().set_prop_cycle(None)
    for label in ['g', 'q', 'w', 'z', 't']:
        y = data1['auc_hls4ml_{}'.format(label)] / baseline['auc_qkeras_{}'.format(label)].iloc[0]
        plt.plot(1, y, 'o')
    for label in ['g', 'q', 'w', 'z', 't']:
        y = data1['auc_hls4ml_{}'.format(label)] / baseline['auc_qkeras_{}'.format(label)].iloc[0]
        plt.plot(1, y, 'o')

    ax1.set_xlim((-0.5, 1.5))
    ax1.get_yaxis().set_visible(False)
    plt.xticks([0, 1], ['A', 'B'])

    plt.savefig('auc.pdf')


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-d', '--dir', action='store', type='string', dest='directory', default='scan_models')
    parser.add_option('-o', '--outfile', action='store', type='string', dest='outfile', default='summary.csv')
    parser.add_option('-i', '--infile', action='store', type='string', dest='infile', default=None)
    (options,args) = parser.parse_args()
     
    outfile = options.directory + options.outfile
    if options.infile is None:
        data = extract_data(options.directory)
        data.to_csv(outfile)
    else:
        data = pandas.read_csv(options.pandasFile)

    make_plots(data)
