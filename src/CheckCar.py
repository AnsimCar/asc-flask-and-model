import torch
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime

from Models import Unet

plt.rcParams['font.family'] = 'Malgun Gothic'

plt.rcParams['axes.unicode_minus'] = False

labels = ['Breakage_3', 'Crushed_2', 'Scratch_0', 'Seperated_1']
models = []

n_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for label in labels:
    model_path = f'models/[DAMAGE][{label}]Unet.pt'

    model = Unet(encoder='resnet34', pre_weight='imagenet',
                 num_classes=n_classes).to(device)
    model.model.load_state_dict(torch.load(
        model_path, map_location=torch.device(device)))
    model.eval()

    models.append(model)

print('Loaded pretrained models!')

def convert(email, img_path):
    path = "converted/"+email
    if not os.path.exists(path):
        os.mkdir(path)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    img_input = img / 255.
    img_input = img_input.transpose([2, 0, 1])
    img_input = torch.tensor(img_input).float().to(device)
    img_input = img_input.unsqueeze(0)

    fig, ax = plt.subplots(1, 5, figsize=(24, 10))

    ax[0].imshow(img)
    ax[0].axis('off')

    outputs = []

    for i, model in enumerate(models):
        output = model(img_input)

        img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
        img_output = img_output.transpose([1, 2, 0])

        select = labels[i]

        if select == 'Breakage_3':
            select = '파손'
        elif select == 'Crushed_2':
            select = '찌그러짐'
        elif select == 'Scratch_0':
            select = '스크래치'
        elif select == 'Seperated_1':
            select = '이격'

        outputs.append(img_output)
        ax[i+1].set_title(select)
        ax[i+1].imshow(img.astype('uint8'), alpha=0.5)
        ax[i+1].imshow(img_output, cmap='jet', alpha=0.5)
        ax[i+1].axis('off')

    fig.set_tight_layout(True)

    converedDir = 'converted/' + email + '/' + email + str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day) + str(datetime.now().hour) + str(datetime.now().minute)
    plt.savefig(converedDir)
    print("saved!")

    return os.getcwd() + '/' + converedDir + '.png'
