from utils.getter import *
import argparse
import pprint


def main(config, args):
    set_seed()
    device = torch.device('cuda' if args.cuda is not None else 'cpu')
    pprint.PrettyPrinter(indent=2).pprint(vars(config))
    trainset = get_instance(
        config.dataset['train'], transforms=transforms_train)
    valset = get_instance(config.dataset['val'],
                          transforms=transforms_val)

    print(trainset)
    print(valset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training custom model')
    parser.add_argument('--cuda', type=bool, default=False, help='Using GPU')
    parser.add_argument('--config', help='yaml config')

    args = parser.parse_args()
    print(args.__dict__)
    config = Config(os.path.join('configs', args.config + '.yaml'))
    main(config, args)
