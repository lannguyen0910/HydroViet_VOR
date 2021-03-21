import yaml


class Hyperparams:
    name = 'HydroViet'
    size = (224, 224)
    batch_size = 16
    num_workers = 2
    lr = 1e-4
    epochs = 20
    pin_memory = True
    log_interval = 20
    log_dir = './ckpt/resnet'
    train_csv = './csv/write_train.csv'
    val_csv = './csv/write_val.csv'
    margin = 1.0
    ckp = f'runs/{name}/model_best.pth.tar'


class Config:

    def __init__(self, yaml_path):
        """
        Config for yaml file
        """
        yaml_file = open(yaml_path)
        self._attr = yaml.load(yaml_file, Loader=yaml.FullLoader)['settings']

    def __getattr__(self, attr):
        try:
            return self._attr[attr]
        except KeyError:
            return None
