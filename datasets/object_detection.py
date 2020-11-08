import torch
import torch.utils.data as data
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from augmentation.transforms import Compose


class ObjectDetectionDataset(data.Dataset):
    """
    - Object detection for COCO dataset Format
    - Bounding Box: (x1, y1, x2, y2)
    - Args:
        + root : directory contains images
        + anno_path: path to annotation-json-file
    """

    def __init__(self, root, anno_path, transforms=None, shuffle=False, max_samples=None):
        self.root = root
        self.anno_path = anno_path
        _, self.ext = os.path.splitext(anno_path)
        self.shuffle = shuffle
        self.transforms = transforms if transforms is not None else Compose()
        self.max_samples = max_samples
        self.data = self.load_annos()
        self.imgs = self.load_images()

    def load_annos(self):
        """
        Load data from annotation file
        """
        data = None
        with open(self.anno_path, 'r') as file:
            if self.ext == '.json':
                data = json.load(file)

        # Label start at index 0
        if data is not None:
            for anno in data['annotations']:
                anno['category_id'] -= 1

            for anno in data['categories']:
                anno['id'] -= 1

        return data

    def cats_to_idx(self):
        """
        Dictionary of category_idx
        """
        self.cats_idx = {}
        self.idx_cats = {}
        self.categories = []
        for cat in self.data['categories']:
            self.categories.append(cat['name'])
            self.cats_idx[cat['name']] = cat['id']
            self.idx_cats[cat['id']] = cat['name']

    def load_images(self):
        """
        Paths of images (List)
        """
        images_list = [os.path.join(self.root, image['file_name'])
                       for image in self.data['images']]

        if self.shuffle:
            random.shuffle(images_list)
        images_list = images_list[:self.max_samples] if self.max_samples is not None and self.max_samples <= len(
            images_list) else images_list

        return images_list

    def count_freq(self, types=1):
        """
        Count frequencies of categories
        """
        count_dict = {}
        if types == 1:
            for cat in self.categories:
                num_images = sum(
                    [1 for i in self.data['annotations'] if i['category_id'] == self.cats_idx[cat]])
                count_dict[cat] = num_images
        elif types == 2:
            pass

        return count_dict

    def plotting(self, figsize=(12, 12), types=['freqs']):
        """
        Frequencies of Categories distribution
        """
        ax = plt.figure(figsize=figsize)
        if 'freqs' in types:
            count_dict = self.count_freq(types=1)
            plt.title(
                f'Total keys in count_dict: {sum(list(count_dict.values()))}')
            barh = plt.barh(list(count_dict.keys()), list(count_dict.values()), color=[
                            np.random.rand(3,) for _ in range(self.categories)])
            for rect in barh:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_height()/2.0, height,
                         '%d' % int(height), ha='center', va='bottom')

        plt.legend()
        plt.show()

    def visualize(self, img, boxes, categories, figsize=(12, 12)):
        """
        Inference image with bounding boxes and categories
        """
        _, ax = plt.subplots(figsize=figsize)

        # Load image
        ax.imshow(img)

        # create boxes and categories by adding rectanges and texts
        for box, category in zip(boxes, categories):
            color = np.random.rand(3,)
            x, y, w, h = box
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor=color, facecolor='None')
            plt.text(
                x, y - 3, self.idx_cats[category], color=color, fontsize=20)
            ax.add_patches(rect)

        plt.show()

    def collate_fn(self, batch):
        """
        - Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        -  This describes how to combine these tensors of different sizes. We use lists.
        - Note: this need not be defined in this Class, can be standalone.
            + param batch: an iterable of N sets from __getitem__()
            + return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images, boxes, categories = [], [], []

        for b in batch:
            images.append(b['img'])
            boxes.append(b['box'])
            categories.append(b['category'])

        images = torch.stack(images, dim=0)

        # tensor (N, 3, 300, 300), 3 lists of N tensors each
        return {
            'imgs': images,
            'boxes': boxes,
            'categories': categories
        }

    def __len__(self):
        return len(self.imgs())

    def __getitem__(self, idx):
        """
        Get a item (image) by passing an idx
        """
        img_item = self.data['images'][idx]
        img_item_id = img_item['id']
        img_item_name = img_item['file_name']
        img_anno_list = [i for i in list(self.data['annotations'])
                         if i['image_id'] == img_item_id]
        img_path = os.path.join(self.root, img_item_name)
        box = np.floor(np.array([i['bbox'] for i in img_anno_list]))
        category = np.array([i['category_id'] for i in img_anno_list])

        img = Image.open(img_path)
        results = self.transforms(img, box, category)
        img, box, category = results['img'], results['box'], results['category']

        return {
            'img': img,
            'box': box,
            'category': category
        }

    def visualize_image(self, idx=None, figsize=(20, 20)):
        if idx == None:
            idx = random.randint(0, len(self.imgs()))

        item = self.__getitem__(idx)
        img = item['img']
        box = item['box']
        category = item['category']

        # Denormalize in transforms
        results = self.transforms.Denormalize(
            img=img, box=box, category=category)
        img, box, category = results['img'], results['box'], results['category']
        self.visualize(img, box, category, figsize)

    def __str__(self) -> str:
        title = 'Dataset for Object Detection\n\n'
        images = f'Number of image sample: {len(self.imgs)}\n'
        categories = f'Number of category: {len(self.categories)}\n'

        return title + images + categories
