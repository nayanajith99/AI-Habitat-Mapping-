
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolov8s-seg.pt', help='path to model')
parser.add_argument('--config', type=str, default='configs/sentisight.yaml', help='path to config file')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--image-size', type=int, default=256, help='dataset image sizes')
parser.add_argument('--save-path', type=str, default='models/sentisight.pt', help='path to save trained model')


if __name__ == '__main__':

    opt = parser.parse_args()

    model = YOLO(f'{opt.model}')
    model.train(data=f'{opt.config}', epochs=opt.epochs, imgsz=opt.image_size, save=True, verbose=True)
    model.save(f'{opt.save_path}')
