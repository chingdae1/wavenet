from glob import glob
from utils import downsampling
import os
import librosa

data_dir = './laughter/'
output_dir = './laughter_normalized/'
# 65 files in all_files
all_files = glob(os.path.join(data_dir, '*'))

'''
# down sampling. to 16k.
for file in all_files:
    val = file.split('/')
    file_name = val[-1]
    # file_name = file_name.replace('.wav', '')
    print(file_name)
    downsampling(data_dir, file_name, output_dir)
'''


'''
# split audio into 1sec.
duplicated_sample = 4000
for file in all_files:
    y, sr = librosa.load(file, sr=16000)
    val = file.split('/')
    file_name = val[-1]
    file_name = file_name.replace('.wav', '')
    prev = 0
    for i in range(sr, y.shape[0], sr-duplicated_sample):
        interval = y[prev:prev+sr]
        librosa.output.write_wav(output_dir + file_name + '_' + str(i) + '.wav', interval, sr)
        prev = i - duplicated_sample
'''

origin = './laughter_preprocessed/'
new = '/Users/changdae/Desktop/sam/'
all_origin = glob(os.path.join(origin, '*'))
all_new = glob(os.path.join(new, '*'))

for idx in range(len(all_new)):
    val = all_new[idx].split('/')
    file_name = val[-1]
    all_new[idx] = file_name

for idx in range(len(all_origin)):
    val = all_origin[idx].split('/')
    file_name = val[-1]
    file_name = file_name.replace('1_')
    all_origin[idx] = file_name

for file in all_new:
    if not file in all_origin:
        os.system('rm ' + new + file)
        print('rm '+new+file)
'''
TO DO LIST

1. 일단 쓸모없는 데이터 지워준다.
2. magnitude normalize. 구글에 audio normalize 참조.
3. wav 파일 mu_quantize 까지 해줘서 .npy 로 저장해준다.
   이거 해줄 때 16000 패딩해서 혹시 모르는 사이즈 안 맞는 놈들 맞춰준다.
4. data 옮겨서 train 한다. regularization 추가해주자.
   generator 에서 sample_offset 확인!!
   웃음 데이터에 대해서는 offset 적용 안해야 함.
   train 해주기 전에 이전 버전의 train, utils, wavenet 파일 백업 해놓자.
5. 구글 서버 하나 만들어서, 거기서는 여러가지 generation 을 시도해보자.
'''
