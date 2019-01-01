import os

from jinja2 import Template

DIR = os.path.dirname(__file__)


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
    TEMPLATE = os.path.join(DIR, 'templates', 'label_map.config')


class DetectionConfig(Config):
    def __init__(self, **kwargs):
        super(DetectionConfig, self).__init__(**kwargs)
        if self.vars.get('model_type') == 'ssd':
            self.TEMPLATE = os.path.join(DIR, 'templates', 'ssd_object_detection.config')
        elif self.vars.get('model_type') == 'faster':
            self.TEMPLATE = os.path.join(DIR, 'templates', 'faster_object_detection.config')
        elif self.vars.get('model_type') == 'ssd_mobile':
            self.TEMPLATE = os.path.join(DIR, 'templates', 'ssd_mobile_object_detection.config')
