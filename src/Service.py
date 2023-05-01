import CheckCar

def checkCar(img):
    splits = str(img).split('/')
    return CheckCar.convert(img, splits[3], splits[5], splits[6], splits[7], splits[9].split('.')[0], splits[9])
