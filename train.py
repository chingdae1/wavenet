from utils import train_generator, valid_generator
from wavenet import build_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


data_dir = '../VCTK_audio_vector/'
valid_data_dir = '../VCTK_valid_vector/'
input_dim = 256
sample_len = 75000
sample_offset = 8000
epoch = 30
batch_size = 3
train_step = (32317//batch_size) + 1
valid_step = (3590// batch_size) + 1
dilation_factor = [1,2,4,8,16,32,64,128,256,512,
                   1,2,4,8,16,32,64,128,256,512,
                   1,2,4,8,16,32,64,128,256,512]

model = build_model(sample_len, dilation_factor)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=4,
                           min_delta=0.00001,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=3,
                               epsilon=0.00001,
                               mode='min'),
             ModelCheckpoint(monitor='val_loss',
                             filepath="./wavenet_weight.hdf5",
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')]

history = model.fit_generator(generator=train_generator(batch_size, input_dim, data_dir, sample_len, sample_offset),
                              steps_per_epoch=train_step,
                              epochs=epoch,
                              callbacks=callbacks,
                              validation_data=valid_generator(batch_size, input_dim, valid_data_dir, sample_len, sample_offset),
                              validation_steps=valid_step,
                              )

# if __name__ == '__main__':
#     onehot = load_data(file_name, data_path, input_dim)
#     sample_len = onehot.shape[0]
#     onehot = np.reshape(onehot, (1, -1, input_dim))
#     in_w = onehot[:, :-1, :]
#     out_w = onehot[:, 1:, :]
#
#     print(np.shape(onehot))