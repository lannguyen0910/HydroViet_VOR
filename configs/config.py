import yaml


class Hyperparams:
    name = 'HydroViet'
    size = (224, 224)
    batch_size = 6
    num_workers = 4
    lr = 1e-3
    epochs = 20
    pin_memory = True
    log_interval = 20
    log_dir = './ckpt/resnet'
    train_csv = './csv/write_train.csv'
    val_csv = './csv/write_val.csv'


class Config():
    def __init__(self, yaml_path):
        yaml_file = open(yaml_path)
        self._attr = yaml.load(yaml_file, Loader=yaml.FullLoader)['settings']

    def __getattr__(self, attr):
        try:
            return self._attr[attr]
        except KeyError:
            return None
