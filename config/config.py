import os

from jinja2 import Template


class Config(object):
    TEMPLATE = None

    def __init__(self, **kwargs):
        self.vars = kwargs

    def write(self, path):
        with open(self.TEMPLATE, 'r') as input_config:
            template = Template(input_config.read())
            content = template.render(self.vars)
            with open(path, 'w') as output_config:
                output_config.write(content)


class LabelMapConfig(Config):
    TEMPLATE = os.path.join('templates', 'label_map.config')


class DetectionConfig(Config):
    TEMPLATE = os.path.join('templates', 'object_detection.config')
