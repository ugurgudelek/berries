# -*- coding: utf-8 -*-
# @Time   : 5/27/2020 5:45 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : logger.py

import neptune
from pathlib import Path
import pandas as pd
import yaml

from abc import ABCMeta, abstractmethod
from collections import defaultdict


class GenericLogger(metaclass=ABCMeta):

    def __init__(self, root, project_name, experiment_name, params, hyperparams):
        self.root = root
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.params = params
        self.hyperparams = hyperparams

    @abstractmethod
    def log_metric(self, log_name, x, y, timestamp=None):
        pass

    @abstractmethod
    def log_image(self, log_name, x, y, image_name=None, description=None, timestamp=None):
        raise NotImplementedError()

    @abstractmethod
    def log_text(self, log_name, x, y, timestamp=None):
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        raise NotImplementedError()

    def stop(self):
        pass


class MultiLogger(GenericLogger):
    def __init__(self, root, project_name, experiment_name, params, hyperparams):
        super(MultiLogger, self).__init__(root, project_name, experiment_name, params, hyperparams)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.params = params
        self.hyperparams = hyperparams

        self.loggers = [
            LocalLogger(root, project_name, experiment_name, params, hyperparams),
            NeptuneLogger(root, project_name, experiment_name, params, hyperparams),
        ]

    def log_metric(self, log_name, x, y, timestamp=None):
        for logger in self.loggers:
            logger.log_metric(log_name, x, y, timestamp)

    def log_image(self, log_name, x, y, image_name=None, description=None, timestamp=None):
        for logger in self.loggers:
            logger.log_image(log_name, x, y, image_name, description, timestamp)

    def log_text(self, log_name, x, y, timestamp=None):
        for logger in self.loggers:
            logger.log_text(log_name, x, y, timestamp)

    def save(self):
        for logger in self.loggers:
            logger.save()


class LocalLogger(GenericLogger):
    def __init__(self, root, project_name, experiment_name, params, hyperparams):
        super(LocalLogger, self).__init__(root, project_name, experiment_name, params, hyperparams)

        self.experiment_fpath = self.root / 'projects' / project_name / experiment_name

        try:
            self.experiment_fpath.mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"Did you change experiment name? : {self.experiment_fpath}")

        # save params and hyperparams
        with open(self.experiment_fpath / 'params.yaml', 'w') as file:
            yaml.dump({**params, **hyperparams}, file, default_flow_style=False, sort_keys=False)

        self.container = {'metric': defaultdict(list),
                          'image': defaultdict(list),
                          'text': defaultdict(list)}

    def log_metric(self, log_name, x, y, timestamp=None):
        self.container['metric'][log_name].append({'x': x, 'y': y, 'timestamp': timestamp})

    def log_image(self, log_name, x, y, image_name=None, description=None, timestamp=None):
        self.container['image'][log_name].append({'x': x, 'y': y,
                                                  'image_name': image_name,
                                                  'description': description,
                                                  'timestamp': timestamp})

    def log_text(self, log_name, x, y, timestamp=None):
        self.container['text'][log_name].append({'x': x, 'y': y, 'timestamp': timestamp})

    def save(self):
        for type_key, type_dict in self.container.items():
            tpath = (self.experiment_fpath / type_key)
            tpath.mkdir(parents=True, exist_ok=True)
            if type_key == 'metric':
                for metric_name, metric_list in type_dict.items():
                    pd.DataFrame(metric_list).to_csv(tpath / f"{metric_name}.csv", index=False)
            if type_key == 'image':
                for log_name, image_list in type_dict.items():
                    ipath = (tpath / log_name)
                    ipath.mkdir(parents=True, exist_ok=True)
                    for image_dict in image_list:
                        image_dict['y'].save(ipath / f"{image_dict['image_name']}.png")
                # drop image container after successful save operation
                self.container['image'] = defaultdict(list)
            if type_key == 'text':
                for log_name, text_list in type_dict.items():
                    pd.DataFrame(text_list).to_csv(tpath / f"{log_name}.csv", index=False)


class NeptuneLogger(GenericLogger):
    def __init__(self, root, project_name, experiment_name, params, hyperparams):
        super(NeptuneLogger, self).__init__(root, project_name, experiment_name, params, hyperparams)

        neptune.init(project_qualified_name=f'ugurgudelek/{project_name}',
                     backend=neptune.HostedNeptuneBackend())
        self.experiment = neptune.create_experiment(name=experiment_name,
                                                    params={**params, **hyperparams},
                                                    upload_stdout=False,
                                                    send_hardware_metrics=False)

    def log_metric(self, log_name, x, y, timestamp=None):
        self.experiment.log_metric(log_name, x, y, timestamp)

    def log_image(self, log_name, x, y, image_name=None, description=None, timestamp=None):
        self.experiment.log_image(log_name, x, y, image_name, description, timestamp)

    def log_text(self, log_name, x, y, timestamp=None):
        self.experiment.log_text(log_name, x, y, timestamp)

    def save(self):
        pass

    def stop(self):
        self.experiment.stop()


if __name__ == "__main__":
    from PIL import Image

    logger = MultiLogger(root=Path("C:/Users/ugur/Documents/GitHub/ai-framework"),
                         project_name='machining',
                         experiment_name='test2',
                         params={'param1': 1,
                                 'param2': 2},
                         hyperparams={'hyperparam1': '1t',
                                      'hyperparam2': '2t'},
                         )
    for epoch in range(10):
        logger.log_metric(log_name='training_loss', x=epoch, y=epoch + 1)
        logger.log_metric(log_name='validation_loss', x=epoch, y=epoch + 3)
        logger.log_image(log_name='img',
                         x=epoch, y=Image.open(Path('../../dereotu.jpeg')),
                         image_name=f"image-{epoch}")
        logger.log_text(log_name='text', x=epoch, y=f'{epoch} - Some Text')
        logger.save()

    logger.stop()
