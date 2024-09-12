
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='best.pt', help='path to model')
parser.add_argument('--image', type=str, default='image.jpg', help='path to image file')
parser.add_argument('--save-path', type=str, default='result.jpg', help='where to save the result image')
parser.add_argument('--confidence', type=float, default=0.5, help='confidence threshold (0.0 <= x <= 1.0)')


if __name__ == '__main__':

    opt = parser.parse_args()

    model = YOLO(opt.model)

    result = model(opt.image, conf=opt.image)

    result[0].save(opt.save_path)
