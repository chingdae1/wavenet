import os
import struct
import wave
import numpy as np
from glob import glob
from keras.preprocessing.sequence import pad_sequences
import librosa
import random
from scipy.io import wavfile

def load_audio(file_path):
    '''
    # original version : cannot load down sampled data (wav type issue)
    # Read raw wave file.
    raw = wave.open(file_path, 'rb')
    # Get number of channels.
    nchannels = raw.getnchannels()
    # Read frame as much as its number of frames.
    raw_frames = raw.readframes(raw.getnframes())
    # Convert binary chunks to floats(amplitude).
    frames = struct.unpack("%ih" % (raw.getnframes() * nchannels), raw_frames)
    frames = [float(val) / pow(2, 15) for val in frames]
    '''

    sr, data = wavfile.read(file_path)
    data = data.tolist()

    return data

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
        if quantized > 255:
            quantized = 255
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

    one_hot = one_hot.astype(np.uint8)

    return one_hot

def train_generator(train_batch_size, input_dim, data_dir, sample_len):
    # VCTK -> 44257 files
    all_files = glob(os.path.join(data_dir, '*npy'))

    while True:
        for start_idx in range(0, len(all_files), train_batch_size):
            x_batch, y_batch = [], []
            for idx in range(start_idx, start_idx + train_batch_size):
                print('t.g!')
                if idx > len(all_files) - 1:
                    idx = random.randrange(0, len(all_files))
                audio_vector = np.load(all_files[idx])
                if audio_vector.shape[0] > sample_len:
                    audio_vector = audio_vector[:sample_len]
                audio_vector = audio_vector.tolist()
                one_hot = q_to_one_hot(audio_vector, input_dim)
                one_hot = one_hot.astype(np.uint8)
                _in = one_hot[:-1, :]
                _out = one_hot[1:, :]
                x_batch.append(_in)
                y_batch.append(_out)

            x_batch = pad_sequences(x_batch, maxlen=sample_len, padding='post')
            y_batch = pad_sequences(y_batch, maxlen=sample_len, padding='post')
            x_batch = np.asarray(x_batch)
            y_batch = np.asarray(y_batch)

            yield x_batch, y_batch

def valid_generator(valid_batch_size, input_dim, valid_data_dir, sample_len):
    # VCTK -> 44257 files
    all_files = glob(os.path.join(valid_data_dir, '*npy'))

    while True:
        for start_idx in range(0, len(all_files), valid_batch_size):
            x_batch, y_batch = [], []
            for idx in range(start_idx, start_idx + valid_batch_size):
                if idx > len(all_files) - 1:
                    idx = random.randrange(0, len(all_files))
                audio_vector = np.load(all_files[idx])
                if audio_vector.shape[0] > sample_len:
                    audio_vector = audio_vector[:sample_len-1]
                audio_vector = audio_vector.tolist()
                one_hot = q_to_one_hot(audio_vector, input_dim)
                one_hot = one_hot.astype(np.uint8)
                _in = one_hot[:-1, :]
                _out = one_hot[1:, :]
                x_batch.append(_in)
                y_batch.append(_out)

            x_batch = pad_sequences(x_batch, maxlen=sample_len, padding='post')
            y_batch = pad_sequences(y_batch, maxlen=sample_len, padding='post')
            x_batch = np.asarray(x_batch)
            y_batch = np.asarray(y_batch)

            yield x_batch, y_batch

def load_generator(all_files):
    for file in all_files:
        file_name = file.split('/')
        file_name = file_name[-1].replace('.wav', '')
        frames = load_audio(file)
        mu_q = mu_quantize(frames, 256)
        mu_q = np.asarray(mu_q)

        yield file_name, mu_q

def save_wav_to_arr(data_dir):
    # VCTK -> 44257 files
    all_files = glob(os.path.join(data_dir, '*wav'))
    output_dir = '../VCTK_audio_vector/'

    for file_name, audio_vector in load_generator(all_files):
        np.save(output_dir + file_name, audio_vector)
        print('save file : ' + file_name)

    print('save_wav_to_arr done.')

def downsampling(data_dir, file_name, downsample_output_dir):
    y, sr = librosa.load(data_dir+file_name, sr=16000)
    librosa.output.write_wav(downsample_output_dir+file_name, y, sr)

def make_valid_set(data_dir):
    all_files = glob(os.path.join(data_dir, '*npy'))
    all_files = all_files[:len(all_files)//10]

    for file in all_files:
        os.system('mv ' + file + ' /home/ubuntu/VCTK_valid_vector')

if __name__ == '__main__':
    data_dir = '../down_VCTK/'

    # data_dir = './data_downsampling/'

    # save_wav_to_arr(data_dir)
    # sample = './data/'
    # file_name = 'p227_001.wav'
    # downsample_output_dir = './data_downsampling/'
    #
    # all_files = glob(os.path.join(data_dir, 'wav48/*/*wav'))
    # for file in all_files:
    #     file_name = file.split('/')[-1]
    #     downsampling(data_dir, file_name, downsample_output_dir)
    #     print('downsampling complete : ' + file)

    save_wav_to_arr(data_dir)
    print('save wav to arr done.')

    data_dir = '../VCTK_audio_vector/'
    make_valid_set(data_dir)


'''
TODO list
오디오 다운샘플링 끝나면 save_wav_to_arr 에 파일 경로 같은거 수정해서
다운샘필링된 놈들 mu_quantize() 까지만 해줘서 넘파이 어레이로 저장

위의 것들 서버에서 돌리는 동안 train_generator 작성
- train_generator 에서는 저장된 npy 부르고 one_hot 으로 바꿔준 다음에
  pad_sequences 해준 다음에 진행해야함
- tolist() 메쏘드로 바꾸고 q_to_one_hot 에 넣어줘야 함

다 되면 train
train 하는 동안 generation 코드 작성

다 되면 generate!

아 근데 그럼 나중에 컨디션 줘서 학습시킬때 지금 pad_sequence 에 max_len 때문에
오디오 잘리는거 주의해줘야 되겠구나
'''