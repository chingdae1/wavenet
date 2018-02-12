from utils import q_to_one_hot, sample, onehot_to_wave
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from wavenet import build_model
import pickle
from scipy.io import wavfile

sr = 16000
sec = 2
sample_len = 30000
input_dim = 256
new_wave_name = './generation/30000_51_p227_03'
audio_file = 'p227_003'
seed_audio_path = '../VCTK_audio_vector/' + audio_file +'.npy'
weight_path = './log_and_weight/30000_4_non_offset.hdf5'
wavfile = '../down_VCTK/' + audio_file + '.wav'
dilation_factor = [1,2,4,8,16,32,64,128,256,512,
                   1,2,4,8,16,32,64,128,256,512,
                   1,2,4,8,16,32,64,128,256,512]

generated_sample = np.load(seed_audio_path)
generated_sample = generated_sample.tolist()
generated_sample = q_to_one_hot(generated_sample, input_dim).astype(np.uint8)
sample_list = []
sample_list.append(generated_sample)
sample_list = pad_sequences(sample_list, maxlen=sample_len, padding='post')
generated_sample = sample_list[0]
generated_sample = np.reshape(generated_sample, (1, sample_len, input_dim))
pred_seed = np.reshape(generated_sample, (-1, input_dim))

model = build_model(sample_len, dilation_factor)
model.load_weights(weight_path)

generation_step = sr * sec

for i in range(generation_step):
    preds = model.predict(np.expand_dims(pred_seed, 0))  # prediction with the model
    sampled = sample(preds[0][-1])  # multinomial sampling
    sampled_onehot = np.zeros([1, 1, input_dim])
    sampled_onehot[0][0][sampled] = 1  # make the sample into onehot
    generated_sample = np.append(generated_sample, sampled_onehot, axis=1)  # append generated sample
    pred_seed = generated_sample[0][i + 1:i + 1 + sample_len]  # make new seed as generation input
    print('generated %ith sample' % (i + 1), end='\r')

# Save generated samples as a flie
with open(new_wave_name+'.pkl', 'wb') as f:
    pickle.dump(generated_sample, f)

# Make an audio file from generated results, generated_{sample_len}_{acc}.wav
quantized = onehot_to_wave(generated_sample, input_dim)
quantized = np.asarray(quantized)
wavfile.write(new_wave_name+'.wav', sr, quantized)

