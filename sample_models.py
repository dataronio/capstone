from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Conv2D, Dense, Reshape, Input, 
    TimeDistributed, Activation, Bidirectional, Maximum, Dropout, Concatenate, MaxPooling1D, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layers, each with batch normalization
    for i in range(recur_layers):
        if i==0:
            name = 'rnn' + str((i+1))
            rnn = GRU(units, activation='relu',
                      return_sequences=True, implementation=2, name=name)(input_data)
            # Add batch normalization
            name = 'bn_' + name
            bn_rnn = BatchNormalization(name=name)(rnn)
        else:
            # Add another recurrent layer
            name = 'rnn' + str((i+1))
            rnn = GRU(units, activation='relu',
                      return_sequences=True, implementation=2, name=name)(bn_rnn)
            # Add batch normalization 
            name = 'bn_' + name
            bn_rnn = BatchNormalization(name=name)(rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
                                  return_sequences=True, implementation=2, name='bidir_rnn'))(input_data)
    bn_bidir = BatchNormalization(name='bn_bidir')(bidir_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_bidir)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def new_cnn_model(input_dim, filters, conv_stride,
                  conv_border_mode, units, output_dim=29):
    """ Build a multiple 1D Convolutional model using merge + maxpooling using spectrogram
    """
    kernels = [11,9] # [11,9,7,5]
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d_0 = Conv1D(filters, kernels[0], 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv_1d_0')(input_data)
    # Add batch normalization
    bn_cnn_0 = BatchNormalization(name='bn_conv_1d_0')(conv_1d_0)
    # now pooling two at time
    #max_1d_0 = MaxPooling1D(pool_size=2, strides=1)(bn_cnn_0)
    # add our second kernel size
    conv_1d_1 = Conv1D(filters, kernels[1], 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_1')(input_data)
    bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)
    #conv_1d_2 = Conv1D(filters, kernels[2], 
    #                 strides=conv_stride, 
    #                 padding=conv_border_mode,
    #                 activation='relu',
    #                 name='conv1d_2')(input_data)
    #bn_cnn_2 = BatchNormalization(name='bn_conv_1d_2')(conv_1d_2)
    #conv_1d_3 = Conv1D(filters, kernels[3], 
    #                 strides=conv_stride, 
    #                 padding=conv_border_mode,
    #                 activation='relu',
    #                 name='conv1d_3')(input_data)
    #bn_cnn_3 = BatchNormalization(name='bn_conv_1d_3')(conv_1d_3)
    #max_1d_1 = MaxPooling1D(pool_size=2, strides=1)(bn_cnn_1)
    concat = Concatenate(axis=1)
    cat = concat([bn_cnn_1, bn_cnn_0])
    # cat = concat([bn_cnn_3, bn_cnn_2, bn_cnn_1, bn_cnn_0])
    #cat = concat([max_1d_1, max_1d_0])
    # Add a recurrent layer
    rnn_units = units # 2 * units
    simp_rnn = SimpleRNN(rnn_units, activation='relu',
                         return_sequences=True, implementation=2, name='rnn')(cat)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # add another SimpleRNN layer
    simp_rnn_2 = SimpleRNN(rnn_units, activation='relu',
                         return_sequences=True, implementation=2, name='rnn_2')(bn_rnn)
    bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(simp_rnn_2)
    # add another TimeDistributed(Dense(50)) layer
    time_dense_2 = TimeDistributed(Dense(50), name='time_dense_2')(bn_rnn_2)
    bn_time_dense_2 = BatchNormalization(name='bn_time_dense_2')(time_dense_2)
    drop_time_dense_2 = Dropout(.4, name='dropout_time_dense_2')(bn_time_dense_2)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(drop_time_dense_2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: new_cnn_output_length(
        x, kernels, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def new_cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    value = 0
    for i in filter_size:
        dilated_filter_size = i + (i - 1) * (dilation - 1)
        if border_mode == 'same':
            output_length = input_length
        elif border_mode == 'valid':
            output_length = input_length - dilated_filter_size + 1
        value = ((output_length + stride - 1) // stride) ## - 1 # uncomment for pooling
    return value

def model9_model(input_dim, filters, conv_stride,
                 conv_border_mode, units, output_dim=29):
    kernels = [11]
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d_0 = Conv1D(filters, kernels[0], 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv_1d_0')(input_data)
    # Add batch normalization
    bn_cnn_0 = BatchNormalization(name='bn_conv_1d_0')(conv_1d_0)
    rnn_units = units # 2 * units
    simp_rnn = SimpleRNN(rnn_units, activation='relu',
                         return_sequences=True, implementation=2, name='rnn')(bn_cnn_0)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # add another SimpleRNN layer
    simp_rnn_2 = SimpleRNN(rnn_units, activation='relu',
                         return_sequences=True, implementation=2, name='rnn_2')(bn_rnn)
    bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(simp_rnn_2)
    # add another TimeDistributed(Dense(50)) layer
    time_dense_2 = TimeDistributed(Dense(50), name='time_dense_2')(bn_rnn_2)
    bn_time_dense_2 = BatchNormalization(name='bn_time_dense_2')(time_dense_2)
    drop_time_dense_2 = Dropout(.4, name='dropout_time_dense_2')(bn_time_dense_2)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(drop_time_dense_2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: new_cnn_output_length(
        x, kernels, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def model10_model(input_dim, filters, conv_stride,
                 conv_border_mode, units, output_dim=29):
    kernels = [11]
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d_0 = Conv1D(filters, kernels[0], 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv_1d_0')(input_data)
    # Add batch normalization
    bn_cnn_0 = BatchNormalization(name='bn_conv_1d_0')(conv_1d_0)
    rnn_units = units # 2 * units
    simp_rnn = SimpleRNN(rnn_units, activation='relu',
                         return_sequences=True, implementation=2, name='rnn')(bn_cnn_0)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # add another SimpleRNN layer
    simp_rnn_2 = SimpleRNN(rnn_units, activation='relu',
                         return_sequences=True, implementation=2, name='rnn_2')(bn_rnn)
    bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(simp_rnn_2)
    # add another TimeDistributed(Dense(100)) layer
    time_dense_1 = TimeDistributed(Dense(100), name='time_dense_1')(bn_rnn_2)
    bn_time_dense_1 = BatchNormalization(name='bn_time_dense_1')(time_dense_1)
    drop_time_dense_1 = Dropout(.5, name='dropout_time_dense_1')(bn_time_dense_1)
    # add another TimeDistributed(Dense(50)) layer
    time_dense_2 = TimeDistributed(Dense(50), name='time_dense_2')(drop_time_dense_1)
    bn_time_dense_2 = BatchNormalization(name='bn_time_dense_2')(time_dense_2)
    drop_time_dense_2 = Dropout(.5, name='dropout_time_dense_2')(bn_time_dense_2)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(drop_time_dense_2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: new_cnn_output_length(
        x, kernels, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def conv2D_model(input_dim, filters, conv_stride,
                 conv_border_mode, units, output_dim=29):
    kernel_size = 11
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim,1))
    # Add convolutional layer
    conv_2d_0 = Conv2D(filters, (11, 11), 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv_2d_0')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_2d_0')(conv_2d_0)
    
    res_out = Reshape((-1, 200))(bn_cnn)
    simp_rnn = SimpleRNN(units, activation='relu',
                         return_sequences=True, implementation=2, name='rnn')(res_out)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def two_input_model(input_dim, filters, conv_stride,
                  conv_border_mode, units, output_dim=29):
    kernels = [11]
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d_0 = Conv1D(filters, kernels[0], 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv_1d_0')(input_data)
    # Add batch normalization
    bn_cnn_0 = BatchNormalization(name='bn_conv_1d_0')(conv_1d_0)
    rnn_units = units # 2 * units
    simp_rnn = SimpleRNN(rnn_units, activation='relu',
                         return_sequences=True, implementation=2, name='rnn')(bn_cnn_0)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # add another SimpleRNN layer
    simp_rnn_2 = SimpleRNN(rnn_units, activation='relu',
                         return_sequences=True, implementation=2, name='rnn_2')(bn_rnn)
    bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(simp_rnn_2)
    # add another TimeDistributed(Dense(50)) layer
    time_dense_2 = TimeDistributed(Dense(50), name='time_dense_2')(bn_rnn_2)
    bn_time_dense_2 = BatchNormalization(name='bn_time_dense_2')(time_dense_2)
    drop_time_dense_2 = Dropout(.4, name='dropout_time_dense_2')(bn_time_dense_2)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(drop_time_dense_2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: new_cnn_output_length(
        x, kernels, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model(input_dim, filters, conv_stride,
                 conv_border_mode, units, output_dim=29):
    """ Build a deep network for speech 
    """
    kernels = [11]
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Specify the layers in your network
    # Add convolutional layer
    conv_1d_0 = Conv1D(filters, kernels[0], 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv_1d_0')(input_data)
    # Add batch normalization
    bn_cnn_0 = BatchNormalization(name='bn_conv_1d_0')(conv_1d_0)
    rnn_units = units 
    Bd_simp_rnn = Bidirectional(SimpleRNN(rnn_units, activation='relu',
                                return_sequences=True, implementation=2, name='Bd_rnn'))(bn_cnn_0)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(Bd_simp_rnn)
    # add another SimpleRNN layer
    Bd_simp_rnn_2 = Bidirectional(SimpleRNN(rnn_units, activation='relu',
                               return_sequences=True, implementation=2, name='Bd_rnn_2'))(bn_rnn)
    bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(Bd_simp_rnn_2)
    # add another TimeDistributed(Dense(100)) layer
    time_dense_1 = TimeDistributed(Dense(100), name='time_dense_1')(bn_rnn_2)
    bn_time_dense_1 = BatchNormalization(name='bn_time_dense_1')(time_dense_1)
    drop_time_dense_1 = Dropout(.5, name='dropout_time_dense_1')(bn_time_dense_1)
    # add another TimeDistributed(Dense(50)) layer
    time_dense_2 = TimeDistributed(Dense(50), name='time_dense_2')(drop_time_dense_1)
    bn_time_dense_2 = BatchNormalization(name='bn_time_dense_2')(time_dense_2)
    drop_time_dense_2 = Dropout(.5, name='dropout_time_dense_2')(bn_time_dense_2)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(drop_time_dense_2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: new_cnn_output_length(
        x, kernels, conv_border_mode, conv_stride)
    print(model.summary())
    return model