import CheckCar

from tkinter import *
from tkinter.filedialog import *

def checkCar(img):
    print(img)
    splits = str(img).split('/')
    print(splits)
    return CheckCar.convert(img, splits[3], splits[5], splits[6], splits[7], splits[9].split('.')[0], splits[9])


def fileOpen():
    global root, photo
    filename = askopenfilename(
        filetypes=(("모든 그림 파일", "*.jpg;*.jpeg;*.bmp;*.png;*.tif"), ("모든 파일", "*.*")),
    )

    CheckCar.convert2(filename)

# fileOpen()