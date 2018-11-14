from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split
from PIL import Image
from functools import partial
import keras.backend as K
import numpy as np
import matplotlib as mtp
mtp.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import glob
import time
import datetime
import argparse
import subprocess

def main():
    learning_way = arguments()
    log_path = make_directories("./log_dir/{0}".format(learning_way))

    if learning_way == "prior":
        learning_data = "./keras_test/"
    elif learning_way == "finetuning":
        learning_data = "./trans_test/"

    Dataset(learning_data, log_path)

    # ml = MachineLearning(learning_data, log_path)
    # ml(learning_way)

def time_counter(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print(
                """\n
activated function : [{0}]
activation time    : {1} [sec]
activation time    : {2} [min]
activation time    : {3} [hour]
                \n""".format(
                    func.__name__,
                    delta,
                    delta/60,
                    delta/3600
                    )
                )
    return wrapper

def make_directories(dir_path):
    if not os.path.exists(dir_path):
        command = "mkdir -p {0}".format(dir_path)
        return_code = subprocess.call(command.split())
        assert return_code == 0,\
                "\nUnix command error...[{0}]\n".format(command)
    return dir_path

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-L","--learning_way",help="learning way",
            required=False, default="prior", choices=["prior","finetuning"]
            )
    parser_args = parser.parse_args()
    learning_way = parser_args.learning_way

    return learning_way

class Dataset(object):
    def __init__(self, learning_data, log_path, height=64, width=64, channel=3):
        self.learning_data = learning_data
        self.dir_list = os.listdir(self.learning_data)
        self.dir_list.sort()
        self.num_label = len(self.dir_list)
        self.height = height
        self.width = width
        self.channel = channel
        self.log_path = log_path

        if self.channel == 1:
            self.mode = "L"
        elif self.channel == 3:
            self.mode = "RGB"
        else:
            raise ValueError("channel is '1' or '3' integer...")

        self.img_predict, self.label_predict,\
                self.img_train, self.label_train,\
                self.img_val, self.label_val = self.data_split()

    # @time_counter
    def data_split(self, val_size=0.2):
        if not os.path.exists(os.path.join(self.log_path, "dataset.npz")):
            img_array, label_array = self.arrangment()
        else:
            data_array = np.load(os.path.join(self.log_path, "dataset.npz"))
            img_array = data_array["x"]
            label_array = data_array["y"]
        img_learning, img_predict, label_learning, label_predict = train_test_split(
                img_array, label_array, test_size=val_size
                )
        img_train, img_val, label_train, label_val = train_test_split(
                img_learning, label_learning, test_size=val_size
                )

        return img_predict, label_predict, img_train, label_train, img_val, label_val

    def arrangment(self):
        img_list = []
        label_list = []
        for dir_name in self.dir_list:
            img_path = os.path.join(self.learning_data, dir_name)
            for i in range(self.num_label):
                for img in glob.glob(os.path.join(img_path, "*.jpg")):
                    if dir_name == self.dir_list[i]:
                        print(os.path.basename(img), i)
                        label_list.append(i)
                        img = Image.open(img).convert(self.mode).resize((self.height, self.width))
                        img = np.asarray(img).astype("float32")/255.
                        img_list.append(img)
        img_array = np.array(img_list)
        label_array = to_categorical(np.array(label_list), self.num_label)
        np.savez(os.path.join(self.log_path, "dataset.npz"), x=img_array, y=label_array)

        return img_array, label_array

class MachineLearning(Dataset):
    def __init__(self, learning_data, log_path, batch_size=32, epochs=100, verbose=0):
        super(MachineLearning, self).__init__(learning_data, log_path)
        self.shape = (self.height, self.width, self.channel)
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

    def __call__(self, learning_way):
        self.model = self.model(learning_way)
        self.results(learning_way)
        self.prediction(learning_way)

    def model(self, learning_way):
        if learning_way == "prior":
            inputs = Input(self.shape)
            x = Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
            x = MaxPooling2D((2,2))(x)
            x = Dropout(0.25)(x)
            x = Conv2D(64, (3,3), padding="same", activation="relu")(x)
            x = Conv2D(64, (3,3), padding="same", activation="relu")(x)
            x = MaxPooling2D((2,2))(x)
            x = Dropout(0.25)(x)
            feature_model = Model(inputs, x)
            x = Flatten()(x)
            x = Dense(512, activation="relu")(x)
            x = Dropout(0.5)(x)
            outputs = Dense(self.num_label, activation="softmax")(x)
            total_model = Model(inputs, outputs)
        elif learning_way == "finetuning":
            prior_model = load_model(os.path.join("./log_dir/prior/prior_model.h5"))

            inputs = Input(prior_model.output_shape[1:])
            x = Flatten()(inputs)
            x = Dense(512, activation="relu")(x)
            x = Dropout(0.5)(x)
            outputs = Dense(self.num_label, activation="softmax")(x)
            top_model = Model(inputs, outputs)
            total_model = Model(prior_model.inputs, top_model(prior_model.outputs))
            for layer in total_model.layers[:5]:
                layer.trainable = False

            # from keras.applications.vgg16 import VGG16
            # prior_model = VGG16(
            #         weights="imagenet", include_top=False,
            #         input_shape=self.shape
            #         )
            #
            # inputs = Input(prior_model.output_shape[1:])
            # x = Flatten()(inputs)
            # x = Dense(512, activation="relu")(x)
            # x = Dropout(0.5)(x)
            # outputs = Dense(self.num_label, activation="softmax")(x)
            # top_model = Model(inputs, outputs)
            # total_model = Model(prior_model.inputs, top_model(prior_model.outputs))
            # for layer in total_model.layers[:15]:
            #     layer.trainable = False

        total_model.compile(
                loss="categorical_crossentropy",
                optimizer=Adam(),
                metrics=self.generate_metrics()
                )
        total_model.summary()
        if learning_way == "prior":
            feature_model.compile(
                    loss="categorical_crossentropy",
                    optimizer=Adam()
                    )
            feature_model.save(
                    os.path.join(self.log_path, "prior_model.h5")
                    )

        return total_model

    @time_counter
    def results(self, learning_way):
        tensorboard = TensorBoard(log_dir=os.path.join(self.log_path, "tflogs"), histogram_freq=1)
        csv_log = CSVLogger(os.path.join(self.log_path, "log.csv"))

        history = self.model.fit(
                self.img_train, self.label_train, batch_size=self.batch_size,
                epochs=self.epochs, verbose=self.verbose,
                validation_data=(self.img_val, self.label_val),
                callbacks=[tensorboard, csv_log]
                )

        key_list = [key for key in history.history.keys() if not "val_" in key]
        for key in key_list:
            self.save_loss_acc(history, key, learning_way)

        score = self.model.evaluate(
                self.img_val, self.label_val, batch_size=self.batch_size,
                verbose=self.verbose
                )
        print(
                """\n
final validation loss     : {0}
final validation accuracy : {1}
                \n""".format(
                    score[0], score[1]
                    )
                )

    @time_counter
    def prediction(self, learning_way):
        predict = self.model.predict(self.img_predict, verbose=self.verbose)

        total = self.img_predict.shape[0]
        counter = 0
        proba_list = []
        with open(os.path.join(self.log_path, "prediction.txt"), "w") as p:
            p.write("***** [{0}] *****\n".format(datetime.datetime.today()))
            for i in range(total):
                proba_list.append(np.amax(predict[i]))
                if np.argmax(self.label_predict[i]) == np.argmax(predict[i]):
                    counter += 1
                p.write(
                        """
true:{0}\tpredicted:{1}\tproba:{2}
                        \n""".format(
                            np.argmax(self.label_predict[i]),
                            np.argmax(predict[i]),
                            predict[i]
                            )
                        )
            p.write("\n***** label-matching *****\n")
            p.write("label-matching : {0} [%]\n".format(counter/total*100))
        self.proba_hist(proba_list, learning_way)

    def normalize_y_pred(self, y_pred):
        return K.one_hot(K.argmax(y_pred), y_pred.shape[-1])

    def class_true_positive(self, class_label, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.cast(K.equal(y_true[:, class_label] + y_pred[:, class_label], 2),
                      K.floatx())

    def class_accuracy(self, class_label, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.cast(K.equal(y_true[:, class_label], y_pred[:, class_label]),
                      K.floatx())

    def class_precision(self, class_label, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.sum(self.class_true_positive(class_label, y_true, y_pred)) / (K.sum(y_pred[:, class_label]) + K.epsilon())

    def class_recall(self, class_label, y_true, y_pred):
        return K.sum(self.class_true_positive(class_label, y_true, y_pred)) / (K.sum(y_true[:, class_label]) + K.epsilon())

    def class_f_measure(self, class_label, y_true, y_pred):
        precision = self.class_precision(class_label, y_true, y_pred)
        recall = self.class_recall(class_label, y_true, y_pred)
        return (2 * precision * recall) / (precision + recall + K.epsilon())

    def true_positive(self, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.cast(K.equal(y_true + y_pred, 2),
                      K.floatx())

    def micro_precision(self, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)
        return K.sum(self.true_positive(y_true, y_pred)) / (K.sum(y_pred) + K.epsilon())

    def micro_recall(self, y_true, y_pred):
        return K.sum(self.true_positive(y_true, y_pred)) / (K.sum(y_true) + K.epsilon())

    def micro_f_measure(self, y_true, y_pred):
        precision = self.micro_precision(y_true, y_pred)
        recall = self.micro_recall(y_true, y_pred)
        return (2 * precision * recall) / (precision + recall + K.epsilon())

    def average_accuracy(self, y_true, y_pred):
        class_count = y_pred.shape[-1]
        class_acc_list = [self.class_accuracy(i, y_true, y_pred) for i in range(class_count)]
        class_acc_matrix = K.concatenate(class_acc_list, axis=0)
        return K.mean(class_acc_matrix, axis=0)

    def macro_precision(self, y_true, y_pred):
        class_count = y_pred.shape[-1]
        return K.sum([self.class_precision(i, y_true, y_pred) for i in range(class_count)]) \
               / K.cast(class_count, K.floatx())

    def macro_recall(self, y_true, y_pred):
        class_count = y_pred.shape[-1]
        return K.sum([self.class_recall(i, y_true, y_pred) for i in range(class_count)]) \
               / K.cast(class_count, K.floatx())

    def macro_f_measure(self, y_true, y_pred):
        precision = self.macro_precision(y_true, y_pred)
        recall = self.macro_recall(y_true, y_pred)
        return (2 * precision * recall) / (precision + recall + K.epsilon())

    def generate_metrics(self):
        metrics = ["accuracy"]

        # the metrics a class label
        func_list = [self.class_accuracy, self.class_precision, self.class_recall, self.class_f_measure]
        name_list = ["acc", "precision", "recall", "f_measure"]
        for i in range(self.num_label):
            for func, name in zip(func_list, name_list):
                func = partial(func, i)
                func.__name__ = "{}-{}".format(name, i)
                metrics.append(func)

        # total metrics
        metrics.append(self.average_accuracy)
        metrics.append(self.macro_precision)
        metrics.append(self.macro_recall)
        metrics.append(self.macro_f_measure)

        return metrics

    def save_loss_acc(self, history, norm, learning_way):
        train_info = history.history[norm]
        val_info = history.history["val_{0}".format(norm)]

        img_name = "{0}_{1}.png".format(learning_way, norm)

        plt.rcParams["font.size"] = 36
        plt.figure(figsize=(20,15))
        plt.plot(train_info, label="train_{0}".format(norm), color="red", lw=5)
        plt.plot(val_info, label="validation_{0}".format(norm), color="green", lw=5)
        plt.xlabel("Learning Steps")
        plt.ylabel("Train {0} and Validation {0}".format(norm))
        plt.title("Learning Curve for {0}".format(norm))
        plt.legend()

        plt.savefig(os.path.join(self.log_path, img_name))

    def proba_hist(self, proba_list, learning_way):
        predict_proba = pd.Series(proba_list)

        plt.rcParams["font.size"] = 36
        plt.figure(figsize=(20,15))
        plt.hist(
                predict_proba, lw=5, bins=100,
                color="green", normed=True, ec="black"
                )
        plt.xlabel("Predicted Probability")
        plt.ylabel("Normalized Frequency")
        plt.title("The Frequency of Predicted Probability")
        plt.savefig(os.path.join(self.log_path, "{0}_hist.png".format(learning_way)))

if __name__ == "__main__":
    main()
