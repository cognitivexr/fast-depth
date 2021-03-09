import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize as imresize

from dataloaders import transforms


def colored_depthmap(depth, d_min=None, d_max=None):
    cmap = plt.cm.viridis

    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)

    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)


class FastDepth:
    def __init__(self):
        self.model = self.load_model()

        self.iheight, self.iwidth = 480, 640  # raw image size
        self.output_size = (224, 224)

        self.transform = transforms.Compose([
            transforms.Resize(250.0 / self.iheight),
            transforms.CenterCrop((228, 304)),
            transforms.Resize(self.output_size),
        ])
        self.to_tensor = transforms.ToTensor()

    def load_model(self):
        torch.nn.Module.dump_patches = True

        folder = './pth'
        # file = 'mobilenet-nnconv5dw-skipadd-pruned.pth.tar'
        file = 'mobilenet-nnconv5dw-skipadd.pth.tar'
        # file = 'mobilenet-nnconv5.pth.tar'

        model_path = os.path.join(folder, file)

        model = torch.load(model_path)['model']
        model = model.cuda()

        return model

    def input_to_tensor(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb_np = self.transform(rgb)
        # rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        input_tensor = self.to_tensor(rgb_np)

        while input_tensor.dim() <= 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.cuda()

        return input_tensor

    def inference(self, img):
        tensor = self.input_to_tensor(img)
        depth = self.model(tensor)
        return depth

    def visualize(self, prediction):
        img = prediction.cpu().numpy()[0, 0]
        img = colored_depthmap(img)

        img = imresize(img, (self.iheight, self.iwidth))

        return np.uint8(img)


def main():
    # download models from http://datasets.lids.mit.edu/fastdepth/results/ into ./pth
    model = FastDepth()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        more, frame = cap.read()
        if not more:
            break

        with torch.no_grad():
            depth = model.inference(frame)

        cv2.imshow('source', frame)
        cv2.imshow('depth', model.visualize(depth))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    print('bye')


if __name__ == '__main__':
    main()
