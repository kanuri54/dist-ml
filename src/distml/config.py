import json 
import logging 
import os
import sys


logger = logging.getLogger(__name__)


class Configuration(dict):

    def __init__(self, config, overrides=None):
        config = self._apply_overrides(config, overrides)
        super().__init__(config)

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __getattr__(self):
        return vars(self)

    def __setstate__(self):
        vars(self).update(state)

    def copy(self):
        return Configuration(json.loads(json.dumps(self)))

    def _apply_overrides(self, d, overrides=None):
        overrides = overrides or {}

        def set_value(d, key, value):
            keys = key.split(".")
            _d = d
            for k in keys[:-1]:
              _d = _d[k]
            _d[keys[-1]] = value

        for key, value in overrides.items():
            set_value(d, key, value)
        return d

    @classmethod
    def from_file(cls, filename, overrides=None):
        logger.info("Reading configuration ...")

        if not os.path.isfile(filename):
            raise ValueError("{} is not a valid configuration path".format(filename))
        else:
            import yaml

            with open(filename) as stream:
                rawconfig = yaml.safe_load(stream)

        config_list = cls(
            rawconfig,
            overrides=overrides
        )   
        
        logger.info("Config parsed.")
        return config_list
