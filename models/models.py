from __future__ import print_function
from tensorflow.keras.layers import Dense, Input,BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l1
import h5py
from constraints import *
from qkeras import *
from qkeras.utils import model_quantize

def three_layer_model(Inputs, nclasses, l1Reg=0):
    """
    Two hidden layers model
    """
    x = Dense(64, kernel_initializer='lecun_uniform', 
              name='fc1', kernel_regularizer=l1(l1Reg))(Inputs)
    x = Activation(activation='relu', name='relu1')(x)
    x = Dense(32, kernel_initializer='lecun_uniform', 
              name='fc2', kernel_regularizer=l1(l1Reg))(x)
    x = Activation(activation='relu', name='relu2')(x)
    x = Dense(32, kernel_initializer='lecun_uniform', 
              name='fc3', kernel_regularizer=l1(l1Reg))(x)
    x = Activation(activation='relu', name='relu3')(x)
    x = Dense(nclasses, kernel_initializer='lecun_uniform', 
                        name='output', kernel_regularizer=l1(l1Reg))(x)
    predictions = Activation(activation='softmax', name='softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def three_layer_model_batch_norm(Inputs, nclasses, l1Reg=0):
    """
    Two hidden layers model
    """
    x = Dense(64, kernel_initializer='lecun_uniform', 
              name='fc1', kernel_regularizer=l1(l1Reg))(Inputs)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn1')(x)
    x = Activation(activation='relu', name='relu1')(x)
              
    x = Dense(32, kernel_initializer='lecun_uniform', 
              name='fc2', kernel_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn2')(x)
    x = Activation(activation='relu', name='relu2')(x)
    
    x = Dense(32, kernel_initializer='lecun_uniform', 
              name='fc3', kernel_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn3')(x)
    x = Activation(activation='relu', name='relu3')(x)
    
    x = Dense(nclasses, kernel_initializer='lecun_uniform', 
                        name='fc4', kernel_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn4')(x)
    predictions = Activation(activation='softmax', name='softmax')(x)

    model = Model(inputs=Inputs, outputs=predictions)
    return model
    
def three_layer_model_apfixed_auto(Inputs, nclasses, l1Reg=0):
    """
    Three hidden layers model
    """
    kmodel = three_layer_model_batch_norm(Inputs, nclasses, l1Reg)
    qconfig = {'default'  : {'kernel_quantizer' : 'quantized_bits(8,0,0)', 'bias_quantizer' : 'quantized_bits(8,0,0)'},
               'fc1_relu' : {'kernel_quantizer' : 'quantized_bits(8,0,0)', 'bias_quantizer' : 'quantized_bits(8,0,0)'},
               'fc2_relu' : {'kernel_quantizer' : 'quantized_bits(8,0,0)', 'bias_quantizer' : 'quantized_bits(8,0,0)'},
               'fc3_relu' : {'kernel_quantizer' : 'quantized_bits(8,0,0)', 'bias_quantizer' : 'quantized_bits(8,0,0)'},
               #'QBatchNormalization' : {'gamma_quantizer'    : 'quantized_bits(16,8)',
               #                         'beta_quantizer'     : 'quantized_bits(16,8)',
               #                         'mean_quantizer'     : 'quantized_bits(16,8)',
               #                         'variance_quantizer' : 'quantized_bits(16,8)'},
                'QActivation' : {'relu' : 'quantized_relu(8)'}
               }
    model = model_quantize(kmodel, qconfig, 8)
    return model

def three_layer_model_opt_0(Inputs, nclasses, l1Reg=0):
    """
    Three hidden layers model
    """


    q_dict = {
     'fc1_relu': {'activation': 'quantized_relu(4,0)',
                  'bias_quantizer': 'quantized_po2(4,8)',
                  'kernel_quantizer': 'quantized_po2(4,1)'
                  },
     'fc2_relu': {'activation': 'quantized_relu(4,0)',
                  'bias_quantizer': 'quantized_po2(4,8)',
                  'kernel_quantizer': "binary(alpha='auto_po2')",
                  },
     'fc3_relu': {'activation': 'quantized_relu(4,0)',
                  'bias_quantizer': 'quantized_bits(4,0,1)',
                  'kernel_quantizer': 'quantized_bits(2,1,1,alpha=1.0)',
                 },
     'output_softmax': {'bias_quantizer': 'quantized_bits(4,0,1)',
                        'kernel_quantizer': "stochastic_binary(alpha='auto_po2')",
                        'units': 5}}
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform',
              name='fc1_relu', kernel_regularizer=l1(l1Reg))(Inputs)
    x = Dense(16, activation='relu', kernel_initializer='lecun_uniform',
              name='fc2_relu', kernel_regularizer=l1(l1Reg))(x)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform',
              name='fc3_relu', kernel_regularizer=l1(l1Reg))(x)
    predictions = Dense(
        nclasses, activation='softmax', kernel_initializer='lecun_uniform',
        name='output_softmax', kernel_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)

    return model_quantize(model, q_dict, 4)

def three_layer_opt_model_1(Inputs, nclasses, l1Reg=0):
    """
    Three hidden layers model
    """
    q_dict = {
    'fc1_relu': {'kernel_quantizer': "quantized_bits(4,0,1,alpha='auto_po2')"},
    'fc2_relu': {'kernel_quantizer': "ternary(alpha='auto_po2')"},
    'fc3_relu': {'kernel_quantizer': 'quantized_bits(2,1,1,alpha=1.0)'},
    'output_softmax': {'bias_quantizer': 'quantized_bits(8,3,1)',
                    'kernel_quantizer': "stochastic_binary(alpha='auto_po2')"},
    'relu1': 'quantized_relu(4,2)',
    'relu2': 'quantized_relu(3,1)',
    'relu3': 'quantized_relu(4,2)'
    }

    x = Dense(32, kernel_initializer='lecun_uniform', use_bias=False,
              name='fc1_relu', kernel_regularizer=l1(l1Reg))(Inputs)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn1')(x)
    x = Activation(activation='relu', name='relu1')(x)
    x = Dense(16, kernel_initializer='lecun_uniform', use_bias=False,
              name='fc2_relu', kernel_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn2')(x)
    x = Activation(activation='relu', name='relu2')(x)
    x = Dense(16, kernel_initializer='lecun_uniform', use_bias=False,
              name='fc3_relu', kernel_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn3')(x)
    x = Activation(activation='relu', name='relu3')(x)
    x = Dense(nclasses, kernel_initializer='lecun_uniform',
              name='output_softmax', kernel_regularizer=l1(l1Reg))(x)
    #x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn4')(x)
    predictions = Activation(activation='softmax', name='softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    model = model_quantize(model, q_dict, 4)
    return model

