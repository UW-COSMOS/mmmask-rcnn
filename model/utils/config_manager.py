import yaml


class Struct:
    def __init__(self, **entries):
        for key, value in entries.items():
            value2 = (Struct(**value) if isinstance(value, dict) else value)
            self.__dict__[key] = value2


class ConfigManager:
    class __Singleton:
        def __init__(self, fp):
            """
            Initialize a singleton config object
            :param fp:
            """
            with open(fp) as fh:
                config = yaml.load(fh)
                for key, value in config.items():
                    value2 = (Struct(**value) if isinstance(value, dict) else value)
                    self.__dict__[key] = value2

    instance = None

    def __init__(self, fp=None):
        if (ConfigManager.instance is None) and (fp is not None):
            ConfigManager.instance = ConfigManager.__Singleton(fp)

    def __getattr__(self, item):
        return getattr(ConfigManager.instance, item)

    def __setattr__(self, key, value):
        setattr(ConfigManager.instance, key, value)
