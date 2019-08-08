import math

from utils import *
from network import *
from config import *
import keras.metrics as metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from Video_Generator import DataGenerator
from keras.optimizers import SGD


def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# gpus = set_visible_devices()
gpus = 1
partition, labels = get_partition_labels()
classes = get_all_classes('classes.txt')

train_generator = DataGenerator(classes=classes, mode=train_file_dir, ids=partition['train'],
                                labels=labels, batch_size=batch_size, clip_length=clip_length, shuffle=True, img_size=img_size)

model_2d = build_2d_net(img_size)
model_3d = build_3d_net(model_2d, num_classes, clip_length, img_size)
if os.path.exists("model_best_weights.h5"):
    model_3d.load_weights("model_best_weights.h5")
    print("从%s中恢复模型权重" % "model_best_weights.h5")
else:
    print("train from scratch")

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='model_best_weights.h5', monitor='loss', verbose=1, save_best_only=True,
                             mode='min', period=1)

lrate = LearningRateScheduler(step_decay)

optimizer = SGD(lr=0.01, momentum=0.9)
if gpus >= 2:
    parallel_model = keras.utils.multi_gpu_model(model_3d, gpus=gpus)
    parallel_model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                           metrics=['accuracy', metrics.top_k_categorical_accuracy])
    parallel_model.fit_generator(generator=train_generator,
                                 epochs=epochs,
                                 callbacks=[checkpoint],
                                 workers=batch_size * 10 * gpus,
                                 max_queue_size=batch_size * 10 * gpus,
                                 use_multiprocessing=True,
                                 shuffle=False)
else:
    model_3d.compile(optimizer=optimizer, loss='categorical_crossentropy',
                     metrics=['accuracy', metrics.top_k_categorical_accuracy])
    model_3d.fit_generator(generator=train_generator,
                           epochs=epochs,
                           callbacks=[checkpoint],
                           workers=batch_size*10,
                           max_queue_size=batch_size*10,
                           use_multiprocessing=True,
                           shuffle=False)
