import yaml


class Config:
    def __init__(self, config: dict):
        for k, v in config.items():
            if isinstance(v, dict):
                v = Config(v)
            if isinstance(v, str):
                try:
                    v = float(v)
                    if v.is_integer():
                        v = int(v)
                except:
                    pass
            config[k] = v

        self.parse_config(config)

    def parse_config(self, config: dict):
        self.__dict__.update(config)


def parse_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return Config(config)
