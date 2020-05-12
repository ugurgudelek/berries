__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from PIL import Image
import pandas as pd
import numpy as np
import os

from PIL import Image
import openpyxl
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from pathlib import Path

def del_all_html():
    [f.unlink() for f in Path('.').glob('**/*.html')]

def plot_confusion_matrix(y_true, y_pred, classes,
                          save_path,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('Blues')
                          ):


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n{cm_norm[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Commented section fixes a bug on some matplotlib version
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] + 0.5, ylim[1] - 0.5)
    fig.tight_layout()
    plt.savefig(save_path)
    return ax


def csv_to_image(csv_path, img_path):
    values = pd.read_csv(csv_path, header=None, sep=' ').values

    values = (values - values.min()) / (values.max() - values.min())
    values *= 255

    img = Image.fromarray(values, mode='L')
    # img = img.resize((224, 224), Image.ANTIALIAS)
    img.save(img_path)
    print(f"image saved : {img_path}")


def changename():
    for filename in os.listdir('.'):
        if filename != 'changename.py':
            kanal = filename.split('_')[0][:5]
            remaining = '_'.join(filename.split('_')[1:])
            remaining = '_' + remaining

            kesimnum = filename.split('_')[0][5:]

            print(kanal, kesimnum, remaining)

            os.rename(filename, f"{kanal}{int(kesimnum)+13*4}{remaining}")


def fill_excelfile(src_filepath, dest_filepath, kind='acc'):
    shutil.copyfile(src_filepath, dest_filepath)
    workbook = openpyxl.load_workbook(dest_filepath)
    worksheet = workbook['Sheet1']

    def read_img_and_insert_excel(img_path, cell_str):
        img = openpyxl.drawing.image.Image(img_path)
        img.height = 300
        img.width = 400
        worksheet.add_image(img, cell_str)

    PATH = '../input/preprocessed_data/alu_v2/kanal{}/00'
    for cutno in range(1, 65):
        cut_path = PATH.format(cutno)
        read_img_and_insert_excel(img_path=os.path.join(cut_path, f'plot_{kind}.png'),
                                  cell_str=f'I{cutno+2}')
        read_img_and_insert_excel(img_path=os.path.join(cut_path, f'plot_fft_{kind}.png'),
                                  cell_str=f'J{cutno+2}')
        read_img_and_insert_excel(img_path=os.path.join(cut_path, f'plot_spectrogram_{kind}.png'),
                                  cell_str=f'K{cutno+2}')

    workbook.save(dest_filepath)


if __name__ == "__main__":
    # csv_to_image(csv_path='../results/fcnn_d06_01stride/49/cm_train.txt',
    #              img_path='../results/fcnn_d06_01stride/49/cm_train.png')

    fill_excelfile(src_filepath='../input/raw_data/alu_v2_verification/alu_v2_parameters_acc.xlsx',
                   dest_filepath='../input/preprocessed_data/alu_v2_verification/alu_v2_parameters_acc.xlsx',
                   kind='acc')
    #


    pass
