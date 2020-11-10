import argparse
import csv
import random
import os
from tqdm import tqdm
#import pprint

parser = argparse.ArgumentParser(description='Split some images')
parser.add_argument('-folder', type=str, default='.',
                    help='read root to folder of images')
parser.add_argument('-ratio', type=float, default=0.8,
                    help='split data to -train and -val set')
parser.add_argument('-new_folder', type=str, default='.',
                    help='path to new_folder (contain train and val folder)')
parser.add_argument('-seed', type=int, default=51,
                    help='seed for splitting process (default = 51)')


args = parser.parse_args()
random.seed(args.seed)

n_classes = os.listdir(args.folder)
for sub in ['train', 'val']:
    path = '_'.join([args.folder, sub])
    if not os.path.exists(path):
        os.mkdir(path)
        for n_class in n_classes:
            new_path = os.path.join(path, n_class)
            os.mkdir(new_path)

num_images, num_train_images, num_val_images = 0, 0, 0
print(f'Splitting folder {args.folder}')
for n_class in n_classes:
    path = os.path.join(args.folder, n_class)
    img_path = list(os.listdir(path))
    random.shuffle(img_path)

    num_image = len(img_path)
    num_images += num_image
    num_train = int(num_image * args.ratio)

    train_imgs = img_path[:num_train]
    val_imgs = img_path[num_train:]

    for img in tqdm(train_imgs):
        num_train_images += 1
        src_path = os.path.join(args.folder, n_class, img)
        tgt_path = os.path.join('_'.join([args.folder, 'train']), n_class, img)

        try:
            os.rename(src_path, tgt_path)
        except Exception:
            print(f'{src_path} not found')

    for img in tqdm(val_imgs):
        num_val_images += 1
        src_path = os.path.join(args.folder, n_class, img)
        tgt_path = os.path.join('_'.join([args.folder, 'val']), n_class, img)

        try:
            os.rename(src_path, tgt_path)
        except Exception:
            print(f'{src_path} not found')

print(f'Number of folder before: {num_images}')
print(f'Number of train folder after: {num_train_images}')
print(f'Number of val folder after: {num_val_images}')
