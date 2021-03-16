from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.getter import *
from torchvision import transforms
from visdom import Visdom
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

import pickle
import argparse
import pandas as pd
import shutil
import numpy as np

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
# from datasets.vocabulary import VocabularyDataset
# from datasets.text_classification import TextClassificationDataset
# from augmentation.nlp_tokenizer import TextTokenizer

# if __name__ == '__main__':
#     dataset = TextClassificationDataset('bbc-text.csv')
#     print(dataset)
#     # dataset.plotting()

#     tokenizer = TextTokenizer(
#         steps=['normal', 'n_grams', 'snowball', 'lemmatize'])

#     vocab = VocabularyDataset(tokenizer=tokenizer, max_length=100000)
#     vocab.build_vocab(dataset)

#     # print(vocab)
#     # print(vocab.freqs)
#     # print(vocab.most_common(50, '2'))
#     # {'the': 50352, 'to': 23856, 'of': 19017, 'and': 17716, 'a': 17518, 'in': 16912}
#     # vocab.plotting(top=10, types=['freqs', '1'])
#     vocab.plotting(top=10, types=['freqs', '3'], figsize=(15, 15))
#     # vocab.plotting(top=10, types=['freqs', '3'])


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    set_seed()

    kwargs = {'num_workers': hp.num_workers,
              'pin_memory': hp.pin_memory} if use_gpu else {}

    model_dict = None
    if hp.ckp is not None:
        if os.path.isfile(hp.ckp):
            print("=> Loading checkpoint '{}'".format(hp.ckp))
            try:
                model_dict = torch.load(hp.ckp)['state_dict']
            except Exception:
                model_dict = torch.load(hp.ckp, map_location='cpu')[
                    'state_dict']
            print("=> Loaded checkpoint '{}'".format(hp.ckp))
        else:
            print("=> No checkpoint found at '{}'".format(hp.ckp))
            return
    else:
        print("Please specify a model")
        return

    model = torch.jit.script(TripletNet(ResNetExtractor(version=152)))
    print('Number of parameters: ', count_parameters(model))

    # model.apply(weights_init)
    model = model.to(device)
    model.load_state_dict(model_dict)

    transform_test = transforms.Compose([
        transforms.Resize(hp.size),
        transforms.CenterCrop(hp.size),
        transforms.ToTensor(),
        normalize
    ])

    testset = TripletDataset(
        root='testing2', transform=transform_test, shuffle=True, mode='test')
    testloader = DataLoader(
        testset, batch_size=hp.batch_size, shuffle=True, **kwargs)

    embeddings = generate_embeddings(testloader, model)

    print(embeddings.shape)
    final_data = {
        'embeddings': embeddings,
    }

    dst_dir = os.path.join('runs', hp.name, 'tSNE')
    make_dir_if_not_exist(dst_dir)

    output_file = open(os.path.join(dst_dir, 'tSNE.pkl'), 'wb')
    pickle.dump(final_data, output_file)
    output_file.close()

    vis_tSNE(embeddings)


def generate_embeddings(data_loader, model):
    with torch.no_grad():
        model.eval()
        labels = None
        embeddings = None
        for batch_idx, batch_imgs in tqdm(enumerate(data_loader)):
            batch_imgs = Variable(batch_imgs.to(device))
            batch_E = model.get_embedding(batch_imgs)
            batch_E = batch_E.data.cpu().numpy()
            embeddings = np.concatenate(
                (embeddings, batch_E), axis=0) if embeddings is not None else batch_E

    return embeddings


def vis_tSNE(embeddings):
    tSNE_ns = 5000
    num_samples = tSNE_ns if tSNE_ns < embeddings.shape[0] else embeddings.shape[0]
    X_embedded = TSNE(n_components=2).fit_transform(
        embeddings[0:num_samples, :])

    fig, ax = plt.subplots()

    x, y = X_embedded[:, 0], X_embedded[:, 1]
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    sc = ax.scatter(x, y, cmap=mpl.colors.ListedColormap(colors))
    plt.colorbar(sc)
    plt.savefig(os.path.join('runs', hp.name, 'tSNE',
                             'tSNE_' + str(num_samples) + '.jpg'))
    plt.show()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    # parser.add_argument('--exp_name', default='exp0', type=str,
    #                     help='name of experiment')
    # parser.add_argument('--cuda', action='store_true', default=False,
    #                     help='enables CUDA training')
    # parser.add_argument('--ckp', default=None, type=str,
    #                     help='path to load checkpoint')
    # parser.add_argument('--dataset', type=str, default='mnist', metavar='M',
    #                     help='Dataset (default: mnist)')

    # parser.add_argument('--pkl', default=None, type=str,
    #                     help='Path to load embeddings')

    # parser.add_argument('--tSNE_ns', default=5000, type=int,
    #                     help='Num samples to create a tSNE visualisation')

    # global args, device
    # args = parser.parse_args()
    # args.cuda = args.cuda and torch.cuda.is_available()
    # cfg_from_file("config/test.yaml")

    main()
