import argparse
import json
import funcy
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description='Split COCO annotations to train_anno and val_anno')

parser.add_argument('--annotations', dest='annotations', metavar='coco_annotations', type=str, default='.',
                    help='path to annotations file')
parser.add_argument('--ratio', dest='ratio', type=float, default=0.8, required=True,
                    help='split data to -train and -val set (0-1)')
parser.add_argument('--annotationed', dest='split_annotations', action='store_true', type=str, default='.',
                    help='Ignore images without annotations.')

args = parser.parse_args()
# Rename
TRAIN = args.annotations[:-5] + '_train.json'
VAL = args.annotations[:-5] + '_val.json'


def save_coco(file, images, annotations, categories):
    with open(file, 'wt', encoding='utf-8') as coco:
        json.dump({'images': images,
                   'annotations': annotations,
                   'categories': categories, }, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def main(args):
    with open(args.annotations, 'rt', encoding='utf-8') as annos:
        coco = json.load(annos)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        images_with_annotations = funcy.lmap(
            lambda a: int(a['image_id']), annotations)

        if args.annotationed:
            images = funcy.lremove(
                lambda i: i['id'] not in images_with_annotations)

        x, y = train_test_split(images, train_size=args.ratio)

        save_coco(TRAIN, x, filter_annotations(annotations, x), categories)
        save_coco(VAL, y, filter_annotations(annotations, y), categories)

        print('Splited!')


if __name__ == '__main__':
    main(args)
