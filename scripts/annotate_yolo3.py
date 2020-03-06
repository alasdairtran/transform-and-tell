"""Detect objects in images.

Based on https://github.com/ultralytics/yolov3/blob/master/detect.py

Usage:
    annotate_yolo3.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    --cfg PATH          *.cfg path [default: tell/yolov3/cfg/yolov3-spp.cfg].
    --names PATH        *.names path [default: tell/yolov3/data/coco.names].
    --weights PATH      Weights path [default: data/yolov3-spp-ultralytics.pt].
    --source PATH       Source [default: data/nytimes/images].
    --output PATH       Output folder [default: data/nytimes/objects].
    --img-size INT      Inference size in pixels [default: 416].
    --conf-thres FLOAT  Object confidence threshold [default: 0.3].
    --iou-thres FLOAT   IOU threshold for NMS [default: 0.6].
    --device DEV        Device ID (i.e. 0 or 0,1) or cpu [default: 0].
    --agnostic-nms      Class-agnostic NMS.
    --dataset DATASET   Dataset [default: nytimes].

"""
import os
import random
import time
from pathlib import Path

import cv2
import ptvsd
import torch
from docopt import docopt
from PIL import Image
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from tell.models.resnet import resnet152
from tell.utils import setup_logger
from tell.yolov3.models import Darknet, attempt_download, load_darknet_weights
from tell.yolov3.utils import torch_utils
from tell.yolov3.utils.datasets import LoadImages
from tell.yolov3.utils.utils import (load_classes, non_max_suppression,
                                     plot_one_box, scale_coords)

logger = setup_logger()
random.seed(123)


def detect(opt):
    if opt['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f"cuda:{opt['device']}")

    resnet = resnet152()
    resnet = resnet.to(device).eval()

    client = MongoClient(host='localhost', port=27017)
    if opt['dataset'] == 'nytimes':
        db = client.nytimes
    elif opt['dataset'] == 'goodnews':
        db = client.goodnews

    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size = opt['img_size']
    out_dir, source, weights = opt['output'], opt['source'], opt['weights']

    # Initialize
    device = torch_utils.select_device(opt['device'])
    os.makedirs(out_dir, exist_ok=True)

    # Initialize model
    model = Darknet(opt['cfg'], img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt['names'])
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, _ in tqdm(dataset):
        if img is None:
            continue

        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pil_image = Image.open(path)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        # We ignore the person class (class 0)
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'],
                                   classes=None, agnostic=opt['agnostic_nms'])

        # Process detections
        assert len(pred) == 1, f'Length of pred is {len(pred)}'
        det = pred[0]
        p, s, im0 = path, '', im0s

        filename = str(Path(p).name)
        basename, ext = os.path.splitext(filename)

        if db.objects.find_one({'_id': basename}):
            continue

        save_path = str(Path(out_dir) / filename)
        s += '%gx%g ' % img.shape[2:]  # print string

        obj_feats = []
        confidences = []
        classes = []
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for j, (*xyxy, conf, class_) in enumerate(det):
                if j >= 64:
                    break
                obj_path = os.path.join(out_dir, f'{basename}_{j:02}.{ext}')

                obj_feat = get_obj_embeddings(
                    xyxy, pil_image, obj_path, resnet)
                obj_feats.append(obj_feat)
                confidences.append(conf.item())
                classes.append(int(class_))

                label = '%s %.2f' % (names[int(class_)], conf)
                plot_one_box(xyxy, im0, label=label,
                             color=colors[int(class_)])

        # Save results (image with detections)
        cv2.imwrite(save_path, im0)

        db.objects.insert_one({
            '_id': basename,
            'object_features': obj_feats,
            'confidences': confidences,
            'classes': classes,
        })

    elapsed = (time.time() - t0) / 3600
    logger.info(f'Done. Results saved to {out_dir}. Object detection takes '
                f'{elapsed:.1f} hours.')


def get_obj_embeddings(xyxy, pil_image, obj_path, resnet):
    pil_image = pil_image.convert('RGB')
    obj_image = extract_object(pil_image, xyxy, save_path=obj_path)

    preprocess = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    obj_image = preprocess(obj_image)
    # obj_image.shape == [n_channels, height, width]

    # Add a batch dimension
    obj_image = obj_image.unsqueeze(0).to(next(resnet.parameters()).device)
    # obj_image.shape == [1, n_channels, height, width]

    X_image = resnet(obj_image, pool=True)
    # X_image.shape == [1, 2048]

    X_image = X_image.squeeze(0).cpu().numpy().tolist()
    # X_image.shape == [2048]

    return X_image


def extract_object(img, box, image_size=224, margin=0, save_path=None):
    """Extract object + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted object image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted object.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin)
    ]
    box = [
        int(max(box[0] - margin[0]/2, 0)),
        int(max(box[1] - margin[1]/2, 0)),
        int(min(box[2] + margin[0]/2, img.size[0])),
        int(min(box[3] + margin[1]/2, img.size[1]))
    ]

    obj = img.crop(box).resize((image_size, image_size), 2)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path)+'/', exist_ok=True)
        save_args = {'compress_level': 0} if '.png' in save_path else {}
        obj.save(save_path, **save_args)

    return obj


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'iou_thres': Use(float),
        'conf_thres': Use(float),
        'img_size': Use(int),
        object: object,
    })
    args = schema.validate(args)
    return args


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)
    os.makedirs('data', exist_ok=True)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    with torch.no_grad():
        detect(args)


if __name__ == '__main__':
    main()
