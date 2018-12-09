import argparse
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from model import get_unet_model, PSNR, L0Loss, UpdateAnnealingParameter
from generator import NoisyImageGenerator, ValGenerator
from noise_model import get_noise_model


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125

def main():
    image_dir = "/home/kjakkala/neuralwave/data/CSI_preprocessed.h5"
    test_dir = "/home/kjakkala/neuralwave/data/CSI_preprocessed.h5"
    batch_size = 8
    nb_epochs = 20
    lr = 0.001
    steps = 1000
    loss_type = "mse"
    output_path = "/home/kjakkala/neuralwave/data"
    model = get_unet_model(1, 1, filters=[16, 16, 32], activation_func='relu', depth=4, inc_rate=2)
    model = multi_gpu_model(model, gpus=4)

    opt = Adam(lr=lr)
    callbacks = []
    source_noise_model = "gaussian,50,0"
    target_noise_model = "clean"

    if loss_type == "l0":
        l0 = L0Loss()
        callbacks.append(UpdateAnnealingParameter(l0.gamma, nb_epochs, verbose=1))
        loss_type = l0()

    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])
    source_noise_model = get_noise_model(source_noise_model)
    target_noise_model = get_noise_model(target_noise_model)
    generator = NoisyImageGenerator(image_dir, source_noise_model, target_noise_model, batch_size=batch_size)
    val_generator = NoisyImageGenerator(test_dir, source_noise_model, target_noise_model, batch_size=batch_size)
    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))

    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
