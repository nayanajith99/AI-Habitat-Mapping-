
from ultralytics import YOLO
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, nargs='+', help='image url for predict')

args = parser.parse_args()

model = YOLO('SentiSightAnnotations/model.pt')
results = model(args.images, conf=0.5)

i = 0
time = str(datetime.datetime.now())
for result in results:
    result.save(f'result{i}-{time}.jpg')
    i += 1
