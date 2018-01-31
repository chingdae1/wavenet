from keras.models import Input, Model
from keras.layers import Conv1D, merge, Activation
from utils import load_data
import numpy as np

def build_model(sample_len, dilation_factor, nb_filters=64, nb_layers=30, input_dim=256):
    def residual_block(x, dilated_rate):
        original_x = x
        tanh_out = Conv1D(2*nb_filters, kernel_size=2, dilation_rate=dilated_rate, padding='causal', activation='tanh')(x)
        sigm_out = Conv1D(2*nb_filters, kernel_size=2, dilation_rate=dilated_rate, padding='causal', activation='sigmoid')(x)
        x = merge(inputs=[tanh_out, sigm_out], mode='mul')

        res_x = Conv1D(nb_filters, kernel_size=1)(x)
        skip_x = Conv1D(nb_filters*4, kernel_size=1)(x)

        res_x = merge(inputs=[original_x, res_x], mode='sum')

        return res_x, skip_x

    audio_in = Input(shape=(sample_len, input_dim))
    out = Conv1D(nb_filters, kernel_size=2, padding='causal', activation='tanh')(audio_in)
    skip_connections = []

    for i in range(nb_layers):
        out, skip_out = residual_block(out, dilation_factor[i])
        skip_connections.append(skip_out)

    out = merge(inputs=skip_connections, mode='sum')
    out = Activation('relu')(out)
    out = Conv1D(input_dim, kernel_size=1, activation='relu')(out)
    out = Conv1D(input_dim, kernel_size=1)(out)

    model = Model(audio_in, out)
    model.summary()

    return model


if __name__ == '__main__':
    data_path = "./data/"
    file_name = "sample.wav"
    input_dim = 256

    onehot = load_data(file_name, data_path, input_dim)
    sample_len = onehot.shape[0]
    onehot = np.reshape(onehot, (1, -1, input_dim))
    in_onehot = onehot[:, :-1, :]
    out_onehot = onehot[:, 1:, :]
    nb_layers = 30

    dilation_factor = [1,2,4,8,16,32,64,128,256,512,
                       1,2,4,8,16,32,64,128,256,512,
                       1,2,4,8,16,32,64,128,256,512]

    build_model(sample_len, dilation_factor)