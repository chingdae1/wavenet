import os
import struct
import wave
import numpy as np
from glob import glob
from keras.preprocessing.sequence import pad_sequences
import librosa


def load_audio(file_path):
    # Read raw wave file.
    raw = wave.open(file_path, 'rb')
    # Get number of channels.
    nchannels = raw.getnchannels()
    # Read frame as much as its number of frames.
    raw_frames = raw.readframes(raw.getnframes())
    # Convert binary chunks to floats(amplitude).
    frames = struct.unpack("%ih" % (raw.getnframes() * nchannels), raw_frames)
    frames = [float(val) / pow(2, 15) for val in frames]

    return frames

def mu_quantize(frames, dim):
    # Set mu. Dimension might be 256.
    mu = dim - 1
    quantized_frames = []

    # Mu-law companding transformation & quantizing
    for x in frames:
        sign_x = np.sign(x)
        log_part = np.log(1 + mu * np.absolute(x)) / np.log(1 + mu)
        f_x = sign_x * log_part
        quantized = int((f_x + 1) / (2 / mu))
        quantized_frames.append(quantized)

    return quantized_frames

# Quantized frames to one-hot vector
def q_to_one_hot(quantized_frames, dim):
    row = np.shape(quantized_frames)[0]
    one_hot = np.zeros((row, dim))
    one_hot[np.arange(row), quantized_frames] = 1

    return one_hot

# Return quantized one-hot vector
def load_data(file_path, dim=256):
    frames = load_audio(file_path)
    mu_q = mu_quantize(frames, dim)
    one_hot = q_to_one_hot(mu_q, dim)

    return one_hot

def train_generator(train_batch_size, input_dim):
    # VCTK -> 44257 files
    all_files = glob(os.path.join(data_dir, 'wav48/*/*wav'))

    # sample.wav

    sample = './data/sample.wav'


    # while True:
    for start_idx in range(0, len(all_files), train_batch_size):
        x_batch, y_batch = [], []
        for idx in range(start_idx, start_idx + train_batch_size):
            if idx > len(all_files) - 1:
                break
            audio_one_hot = load_data(all_files[idx])

            audio_one_hot = np.reshape(audio_one_hot, (train_batch_size, -1, input_dim))
            print(np.shape(audio_one_hot))
            _in = audio_one_hot[:, :-1, :]
            _out = audio_one_hot[:, 1:, :]

            x_batch.append(_in)
            y_batch.append(_out)

        yield x_batch, y_batch


    # for idx in range(len(all_files)):
    #     x_batch, y_batch = [], []
    #     audio_one_hot = load_data(all_files[idx])
    #     audio_one_hot = np.reshape(audio_one_hot, (train_batch_size, -1, input_dim))
    #     _in = audio_one_hot[:, :-1, :]
    #     _out = audio_one_hot[:, 1:, :]
    #
    #     x_batch.append(_in)
    #     y_batch.append(_out)
    #     print(x_batch)
    #
        # yield x_batch, y_batch

def save_wav_to_arr(data_dir):
    # VCTK -> 44257 files
    all_files = glob(os.path.join(data_dir, 'wav48/*/*wav'))

    # For test, get sample
    # all_files = glob(os.path.join(data_dir, '*wav'))

    output_dir = './audio_vector/'
    one_hot_list = []
    file_name_list = []

    for file in all_files:
        file_name = file.split('/')
        file_name = file_name[-1].replace('.wav', '')
        one_hot = load_data(file)
        one_hot_list.append(one_hot)
        file_name_list.append(file_name)
        print('load_data complete : ' + file_name)

    one_hot_list = pad_sequences(one_hot_list, padding='pre')
    i = 0

    for one_hot in one_hot_list:
        np.save(output_dir + file_name_list[i], one_hot)
        print('save file : ' + file_name_list[i])
        i = i + 1

def downsampling(data_dir, file_name, downsample_output_dir):
    y, sr = librosa.load(data_dir+file_name, sr=16000)
    librosa.output.write_wav(downsample_output_dir+file_name, y, sr)

if __name__ == '__main__':
    data_dir = '../VCTK-Corpus/'
    # save_wav_to_arr(data_dir)
    # sample = './data/'
    # file_name = 'p227_001.wav'
    downsample_output_dir = './data_downsampling/'

    all_files = glob(os.path.join(data_dir, 'wav48/*/*wav'))
    for file in all_files:
        file_name = file.split('/')[-1]
        downsampling(data_dir, file_name, downsample_output_dir)
        print('downsampling complete : ' + file)