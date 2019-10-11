"""Get articles from the New York Times API.

Usage:
    detect_faces.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -d --image-dir DIR  Image directory [default: ./data/nytimes/images_processed].
    -f --face-dir DIR   Image directory [default: ./data/nytimes/faces].
    -b --batch INT      Batch number [default: 1]
    -h --host HOST      Mongo host name [default: localhost]

"""

import os
from datetime import datetime

import cv2
import numpy as np
import ptvsd
import torch
from docopt import docopt
from joblib import Parallel, delayed
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from newser.dsfd.data import (TestBaseTransform, WIDERFace_CLASSES,
                              widerface_640)
from newser.dsfd.face_ssd import build_ssd
from newser.utils import setup_logger

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'image_dir': str,
        'face_dir': str,
        'batch': Use(int),
        'host': str,
    })
    args = schema.validate(args)
    return args


def detect_faces(article, nytimes, image_dir, face_dir, net):
    if 'n_images_with_faces' in article:
        return

    sections = article['parsed_section']
    image_positions = article['image_positions']
    visual_threshold = 0.5
    article['face_positions'] = []

    for pos in image_positions:
        section = sections[pos]
        image_path = os.path.join(image_dir, f"{section['hash']}.jpg")
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path} from article "
                         f"{article['_id']} at position {pos}")
            return

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        with torch.no_grad():
            try:
                dets = detect_one_image(img, net)
            except RuntimeError:
                logger.error(f"Fail to parse {image_path} from article "
                             f"{article['_id']} at position {pos}")
                return

        inds = np.where(dets[:, -1] >= visual_threshold)[0]
        section['n_faces'] = len(inds)
        section['faces'] = []
        for i, idx in enumerate(inds):
            x1, y1, x2, y2, score = dets[idx]
            face_path = os.path.join(face_dir, f"{section['hash']}_{i:03}.jpg")
            section['faces'].append({
                'bbox': (x1, y1, x2, y2),
                'score': score,
                'size': (y2 - y1) * (x2 - x1),
            })

            face_crop = img[int(round(y1)):int(round(y2)),
                            int(round(x1)):int(round(x2))]

            cv2.imwrite(face_path, face_crop)

        if section['n_faces'] > 0:
            article['face_positions'].append(pos)

    article['n_images_with_faces'] = len(article['face_positions'])

    nytimes.articles.find_one_and_update(
        {'_id': article['_id']}, {'$set': article})


def detect_one_image(img, net):
    thresh = widerface_640['conf_thresh']
    transform = TestBaseTransform((104, 117, 123))
    cuda = True

    max_im_shrink = ((2000.0*2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    det0 = infer(net, img, transform, thresh, cuda, shrink)
    det1 = infer_flip(net, img, transform, thresh, cuda, shrink)
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = infer(net, img, transform, thresh, cuda, st)
    index = np.where(np.maximum(
        det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    factor = 2
    bt = min(factor, max_im_shrink) if max_im_shrink > 1 else (
        st + max_im_shrink) / 2
    det_b = infer(net, img, transform, thresh, cuda, bt)
    # enlarge small iamge x times for small face
    if max_im_shrink > factor:
        bt *= factor
        while bt < max_im_shrink:
            det_b = np.row_stack(
                (det_b, infer(net, img, transform, thresh, cuda, bt)))
            bt *= factor
        det_b = np.row_stack(
            (det_b, infer(net, img, transform, thresh, cuda, max_im_shrink)))
    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    det = np.row_stack((det0, det1, det_s, det_b))
    det = bbox_vote(det)

    return det


def infer(net, img, transform, thresh, cuda, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink,
                         interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = x.unsqueeze(0)
    if cuda:
        x = x.cuda()
    # print (shrink , x.shape)
    y = net(x)      # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([img.shape[1]/shrink, img.shape[0]/shrink,
                          img.shape[1]/shrink, img.shape[0]/shrink])
    det = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            # label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            det.append([pt[0], pt[1], pt[2], pt[3], score.item()])
            j += 1
    if (len(det)) == 0:
        det = [[0.1, 0.1, 0.2, 0.2, 0.01]]
    det = np.array(det)

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def infer_flip(net, img, transform, thresh, cuda, shrink):
    img = cv2.flip(img, 1)
    det = infer(net, img, transform, thresh, cuda, shrink)
    det_t = np.zeros(det.shape)
    det_t[:, 0] = img.shape[1] - det[:, 2]
    det_t[:, 1] = det[:, 1]
    det_t[:, 2] = img.shape[1] - det[:, 0]
    det_t[:, 3] = det[:, 3]
    det_t[:, 4] = det[:, 4]
    return det_t


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = None
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        if dets is not None:
            dets = np.row_stack((dets, det_accu_sum))
        else:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)
    image_dir = args['image_dir']
    face_dir = args['face_dir']

    os.makedirs(face_dir, exist_ok=True)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    logger.info('Loading model.')
    cfg = widerface_640
    num_classes = len(WIDERFace_CLASSES) + 1  # +1 background
    net = build_ssd('test', cfg['min_dim'], num_classes)  # initialize SSD
    net.load_state_dict(torch.load('dsfd/weights/WIDERFace_DSFD_RES152.pth'))
    net = net.cuda().eval()

    client = MongoClient(host=args['host'], port=27017)
    nytimes = client.nytimes

    if args['batch'] == 1:  # 38K
        start = datetime(2000, 1, 1)
        end = datetime(2007, 8, 1)
    elif args['batch'] == 2:  # 42K
        start = datetime(2007, 8, 1)
        end = datetime(2009, 1, 1)
    elif args['batch'] == 3:  # 41K
        start = datetime(2009, 1, 1)
        end = datetime(2010, 5, 1)
    elif args['batch'] == 4:  # 79K
        start = datetime(2010, 5, 1)
        end = datetime(2012, 11, 1)
    elif args['batch'] == 5:  # 80K
        start = datetime(2012, 11, 1)
        end = datetime(2015, 5, 1)
    elif args['batch'] == 6:  # 80K
        start = datetime(2015, 5, 1)
        end = datetime(2017, 5, 1)
    elif args['batch'] == 7:  # 81K
        start = datetime(2017, 5, 1)
        end = datetime(2019, 9, 1)
    else:
        raise ValueError(f"Unknown batch: {args['batch']}")

    article_cursor = nytimes.articles.find({
        'parsed': True,  # article body is parsed into paragraphs
        'n_images': {'$gt': 0},  # at least one image is present
        'language': 'en',
        'pub_date': {'$gte': start, '$lt': end},
    }, no_cursor_timeout=True).batch_size(128)

    logger.info('Detecting faces.')
    with Parallel(n_jobs=8, backend='threading') as parallel:
        parallel(delayed(detect_faces)(article, nytimes, image_dir, face_dir, net)
                 for article in tqdm(article_cursor))


if __name__ == '__main__':
    main()
