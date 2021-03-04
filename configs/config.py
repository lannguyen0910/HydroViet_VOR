import yaml


class Hyperparams:
    size = (500, 500)
    batch_size = 6
    num_workers = 4
    lr = 1e-3
    epochs = 200
    pin_memory = True


class Config():
    def __init__(self, yaml_path):
        yaml_file = open(yaml_path)
        self._attr = yaml.load(yaml_file, Loader=yaml.FullLoader)['settings']

    def __getattr__(self, attr):
        try:
            return self._attr[attr]
        except KeyError:
            return None
