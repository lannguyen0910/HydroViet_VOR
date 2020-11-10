import argparse
import csv
import random
#import pprint

parser = argparse.ArgumentParser(description='Process some parameters')
parser.add_argument('-csv', type=str, default='test.csv',
                    help='read root to csv file')
parser.add_argument('-ratio', type=float, default=0.8,
                    help='split csv data to -train and -val set')
parser.add_argument('-skip_header', type=str,
                    default=True, help='Skip csv header')
parser.add_argument('-new_csv', type=str, default='.',
                    help='path to new_csv (default is current dir)')
parser.add_argument('-seed', type=int, default=51,
                    help='seed for splitting process (default = 51)')


args = parser.parse_args()

# Create random seed first time
random.seed(args.seed)
with open(args.csv, 'r', encoding='utf8') as f:
    rows = csv.reader(f)
    if args.skip_header:
        next(rows)
    data = list(rows)

train_set = str(args.csv) + '_train'
val_set = str(args.csv) + '_val'

DICT = {
    train_set: {},
    val_set: {}
}

class_dict = {}
for category, text in data:
    class_dict.setdefault(category, [])
    class_dict[category].append(text)

for cls_id, cls in class_dict.items():
    train_size = max(int(len(cls) * args.ratio), 1)
    shuffle = random.sample(cls, k=len(cls))
    DICT[train_set][cls_id] = shuffle[:train_size]
    DICT[val_set][cls_id] = shuffle[train_size:]


# Create train.csv and val.csv from dataset.csv
for dicts, classes in DICT.items():
    new = [['category', 'text']]
    new.extend([
        [cat, text] for cat, texts in classes.items() for text in texts
    ])

    file_write = f'{args.new_csv}/{dicts}.csv'
    with open(file_write, 'w', newline='', encoding='utf8') as w:
        writer = csv.writer(w)
        writer.writerows(new)
