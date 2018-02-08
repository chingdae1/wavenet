from utils import load_data, train_generator
from wavenet import build_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np


data_dir = '../VCTK_audio_vector/'
input_dim = 256
sample_len = 100000
epoch = 30
batch_size = 1
dilation_factor = [1,2,4,8,16,32,64,128,256,512,
                   1,2,4,8,16,32,64,128,256,512,
                   1,2,4,8,16,32,64,128,256,512]

model = build_model(sample_len, dilation_factor)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# callbacks = [EarlyStopping(monitor='val_loss',
#                            patience=4,
#                            min_delta=0.00001,
#                            mode='min'),
#              ReduceLROnPlateau(monitor='val_loss',
#                                factor=0.1,
#                                patience=3,
#                                epsilon=0.00001,
#                                mode='min'),
#              ModelCheckpoint(monitor='val_loss',
#                              filepath="./wavenet_weight.hdf5",
#                              save_best_only=True,
#                              save_weights_only=True,
#                              mode='min')]

train_step = (44257//batch_size) + 1
history = model.fit_generator(generator=train_generator(batch_size, input_dim, data_dir),
                              steps_per_epoch=train_step,
                              epochs=epoch
                              # callbacks=callbacks,
                              )

# if __name__ == '__main__':
#     onehot = load_data(file_name, data_path, input_dim)
#     sample_len = onehot.shape[0]
#     onehot = np.reshape(onehot, (1, -1, input_dim))
#     in_w = onehot[:, :-1, :]
#     out_w = onehot[:, 1:, :]
#
#     print(np.shape(onehot))