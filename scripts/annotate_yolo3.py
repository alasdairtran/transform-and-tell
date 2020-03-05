"""Detect objects in images.

Based on https://github.com/ultralytics/yolov3/blob/master/detect.py

Usage:
    annotate_yolo3.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    --cfg PATH          *.cfg path [default: tell/yolov3/cfg/yolov3-spp.cfg].
    --names PATH        *.names path [default: tell/yolov3/data/coco.names].
    --weights PATH      Weights path [default: data/yolov3-spp-ultralytics.pt].
    --source PATH       Source [default: data/samples].
    --output PATH       Output folder [default: output].
    --img-size INT      Inference size in pixels [default: 416].
    --conf-thres FLOAT  Object confidence threshold [default: 0.3].
    --iou-thres FLOAT   IOU threshold for NMS [default: 0.6].
    --fourcc TYPE       Output video codec (verify ffmpeg support) [default: mp4v].
    --half              Half precision FP16 inference.
    --device DEV        Device ID (i.e. 0 or 0,1) or cpu [default: 0].
    --view-img          Display results.
    --save-txt          Save results to *.txt.
    --classes C...      Filter by class.
    --agnostic-nms      Class-agnostic NMS.

"""
import os
import random
import shutil
import time
from pathlib import Path
from sys import platform

import cv2
import ptvsd
import torch
from docopt import docopt
from schema import And, Or, Schema, Use

from tell.utils import setup_logger
from tell.yolov3.models import Darknet, attempt_download, load_darknet_weights
from tell.yolov3.utils import torch_utils
from tell.yolov3.utils.datasets import LoadImages, LoadStreams
from tell.yolov3.utils.utils import (apply_classifier, load_classes,
                                     non_max_suppression, plot_one_box,
                                     scale_coords)

logger = setup_logger()
random.seed(123)


def detect(opt):
    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size = opt['img_size']
    out, source, weights, half, view_img, save_txt = opt['output'], opt[
        'source'], opt['weights'], opt['half'], opt['view_img'], opt['save_txt']
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt['device'])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt['cfg'], img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(
            name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()
    # torch_utils.model_info(model, report='summary')  # 'full' or 'summary'

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        # set True to speed up constant image size inference
        torch.backends.cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt['names'])
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0].float() if half else model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt['fourcc']), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


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

    detect(args)


if __name__ == '__main__':
    main()
