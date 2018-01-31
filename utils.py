import os
import struct
import wave
import numpy as np
from glob import glob

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
    data_dir = '../data/VCTK-Corpus/'

    # VCTK -> 44257 files
    all_files = glob(os.path.join(data_dir, 'wav48/*/*wav'))

    # sample.wav

    sample = './data/sample.wav'

    # batch size more than 1 -> fail
    # while True:
    # for start_idx in range(0, len(all_files), train_batch_size):
    #     x_batch, y_batch = [], []
    #     for idx in range(start_idx, start_idx + train_batch_size):
    #         if idx > len(all_files) - 1:
    #             break
    #         audio_one_hot = load_data(all_files[idx])
    #
    #         audio_one_hot = np.reshape(audio_one_hot, (train_batch_size, -1, input_dim))
    #         print(np.shape(audio_one_hot))
    #         _in = audio_one_hot[:, :-1, :]
    #         _out = audio_one_hot[:, 1:, :]
    #
    #         x_batch.append(_in)
    #         y_batch.append(_out)
    #
    #     yield x_batch, y_batch


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
    while True:
        x_batch, y_batch = [], []
        audio_one_hot = load_data(sample)
        audio_one_hot = np.reshape(audio_one_hot, (train_batch_size, -1, input_dim))
        _in = audio_one_hot[:, :-1, :]
        _out = audio_one_hot[:, 1:, :]

        x_batch.append(_in)
        y_batch.append(_out)


        # yield x_batch, y_batch
if __name__ == '__main__':
    train_generator(1, 256)
    print('hi')
