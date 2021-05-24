# -*- coding: utf-8 -*-
# @Time   : 5/27/2020 5:45 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : logger.py

from pathlib import Path
import pandas as pd
import yaml

from abc import ABCMeta, abstractmethod
from collections import defaultdict


class GenericLogger(metaclass=ABCMeta):

    def __init__(self, root, project_name, experiment_name, params,
                 hyperparams):
        self.root = root
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.params = params
        self.hyperparams = hyperparams

    @abstractmethod
    def log_history(self, phase, epoch, history):
        pass

    @abstractmethod
    def log_metric(self,
                   metric_name,
                   phase,
                   epoch,
                   metric_value,
                   timestamp=None):
        pass

    @abstractmethod
    def log_image(self,
                  img,
                  image_name=None,
                  description=None,
                  timestamp=None,
                  **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def log_text(self, text, timestamp=None, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        raise NotImplementedError()

    def log_dataframe(self, key, dataframe):
        pass

    def log_model(self, path, name):
        pass

    def stop(self):
        pass


class MultiLogger(GenericLogger):

    def __init__(self,
                 root,
                 project_name,
                 experiment_name,
                 params,
                 hyperparams,
                 offline=False):
        super(MultiLogger, self).__init__(root, project_name, experiment_name,
                                          params, hyperparams)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.params = params
        self.hyperparams = hyperparams
        self.offline = offline

        self._loggers = {
            'local':
                LocalLogger(root, project_name, experiment_name, params,
                            hyperparams),
        }
        if not offline:
            self._loggers['neptune'] = NeptuneLogger(root, project_name,
                                                     experiment_name, params,
                                                     hyperparams)

    def log_metric(self,
                   metric_name,
                   phase,
                   epoch,
                   metric_value,
                   timestamp=None):
        for logger_name, logger in self._loggers.items():
            logger.log_metric(metric_name, phase, epoch, metric_value,
                              timestamp)

    def log_history(self, phase, epoch, history):
        raise NotImplementedError()

    def log_image(self,
                  img,
                  image_name=None,
                  description=None,
                  timestamp=None,
                  **kwargs):
        for logger_name, logger in self._loggers.items():
            logger.log_image(img, image_name, description, timestamp, **kwargs)

    def log_text(self, text, timestamp=None, **kwargs):
        for logger_name, logger in self._loggers.items():
            logger.log_text(text, timestamp=None, **kwargs)

    def log_dataframe(self, key, dataframe):
        for logger_name, logger in self._loggers.items():
            logger.log_dataframe(key, dataframe)

    def log_model(self, path, name='model'):
        for logger_name, logger in self._loggers.items():
            logger.log_model(path, name)

    def save(self):
        for logger_name, logger in self._loggers.items():
            logger.save()

    def stop(self):
        for logger_name, logger in self._loggers.items():
            logger.stop()


class LocalLogger(GenericLogger):

    def __init__(self, root, project_name, experiment_name, params,
                 hyperparams):
        super(LocalLogger, self).__init__(root, project_name, experiment_name,
                                          params, hyperparams)

        self.experiment_fpath = self.root.absolute() / 'projects' / \
            project_name / experiment_name

        try:
            self.experiment_fpath.mkdir(parents=True)
        except FileExistsError:
            if 'debug' == project_name:
                print("Logger running because of debug keyword.")
            elif params['pretrained'] or params['resume']:
                print(
                    "Logger starting from existing directory because of pretrained or resume keyword."
                )
            else:
                raise FileExistsError(
                    f"Did you change experiment name? : {self.experiment_fpath}"
                )

        # save params and hyperparams
        with open(self.experiment_fpath / 'params.yaml', 'w') as file:
            pa = params.copy()
            pa['root'] = str(pa['root'].absolute())

            yaml.dump({
                **pa,
                **hyperparams
            },
                      file,
                      default_flow_style=False,
                      sort_keys=False)

        self.container = {
            'metric': defaultdict(lambda: defaultdict(list)),
            'image': list(),
            'text': list(),
            'history': defaultdict(dict)
        }

    def log_metric(self,
                   metric_name,
                   phase,
                   epoch,
                   metric_value,
                   timestamp=None):
        self.container['metric'][phase][metric_name].append({
            'epoch': epoch,
            'value': metric_value
        })

    def log_history(self, phase, epoch, history):
        self.container['history'][phase][epoch] = history

    def log_image(self,
                  img,
                  image_name=None,
                  description=None,
                  timestamp=None,
                  **kwargs):
        # self.container['image'].append({
        #     'img': img,
        #     'image_name': image_name,
        #     'description': description,
        #     **kwargs, 'timestamp': timestamp
        # })
        pass

    def log_text(self, text, timestamp=None, **kwargs):
        self.container['text'].append({
            'text': text,
            **kwargs, 'timestamp': timestamp
        })

    def save(self):
        for ckey, cval in self.container.items():
            tpath = (self.experiment_fpath / ckey)
            if cval:
                tpath.mkdir(parents=True, exist_ok=True)

            if ckey == 'history':
                for phase, epoch_dict in cval.items():
                    for epoch, history_dict in epoch_dict.items():
                        # for each history_dict's key
                        # there will be a ndarray of corresponding history item
                        # so we need to transform ndarray to 1d array
                        d = dict()
                        for key, ndarr in history_dict.items():
                            if ndarr.ndim == 0:
                                raise Exception(
                                    'ndarr dim should be at least 1')
                            elif ndarr.ndim == 1:
                                d[key] = ndarr
                            elif ndarr.ndim == 2:
                                for col_ix in range(ndarr.shape[1]):
                                    d[f'{key}_{col_ix}'] = ndarr[:, col_ix]

                        pd.DataFrame(d).to_csv(tpath /
                                               f'{phase}-{epoch}-history.csv')

                # reset container
                self.container['history'] = defaultdict(dict)

            if ckey == 'metric':  # cval: dict[phase]][metric_name]
                for phase, metric_dict in cval.items(
                ):  # metric_dict:  key: metric_name, val: [{'epoch', 'metric_Value'}]

                    metrics = list()
                    for metric_name, metric_list in metric_dict.items():
                        metric_df = pd.DataFrame(metric_list).set_index(
                            'epoch').rename(columns={'value': metric_name})
                        metrics.append(metric_df)

                    write_path = tpath / f'{phase}-metric.csv'
                    mode, header = ('a', False) if write_path.exists() else ('w', True) # yapf:disable

                    pd.concat(metrics, axis=1).to_csv(write_path,
                                                      mode=mode,
                                                      header=header)

                    # reset container
                    self.container['metric'] = defaultdict(lambda: defaultdict(list)) # yapf:disable

            if ckey == 'image':
                for item_dict in cval:
                    item_dict['img'].save(tpath /
                                          f"{item_dict['image_name']}.png")
                # drop image container after successful save operation
                self.container['image'] = defaultdict(list)
            if ckey == 'text':
                if cval:  # if it has any element
                    pd.DataFrame(cval).to_csv(tpath / "text.csv", index=False)


class NeptuneLogger(GenericLogger):

    # todo: test log_metric, log_image and log_text methods

    def __init__(self, root, project_name, experiment_name, params,
                 hyperparams):
        super(NeptuneLogger, self).__init__(root, project_name, experiment_name,
                                            params, hyperparams)
        import neptune.new as neptune
        from neptune.new.types import File

        self._File = File

        neptune_params = params['neptune']
        workspace = neptune_params['workspace']
        project = neptune_params['project']
        source_files = neptune_params['source_files']

        run_id = neptune_params.get('id', False)
        if run_id:
            self.run = neptune.init(project=f'{workspace}/{project}',
                                    run=run_id,
                                    source_files=source_files)
        else:
            self.run = neptune.init(project=f'{workspace}/{project}',
                                    source_files=source_files)

        self.run['sys/tags'].add(neptune_params['tags'])

        self.run['parameters'] = params
        self.run['hyperparameters'] = hyperparams

    def log_metric(self,
                   metric_name,
                   phase,
                   epoch,
                   metric_value,
                   timestamp=None):

        self.run[f'{phase}/{metric_name}'].log(metric_value, step=epoch)

    def log_history(self, phase, epoch, history):
        raise NotImplementedError()

    def log_image(self,
                  img,
                  image_name=None,
                  description=None,
                  timestamp=None,
                  **kwargs):

        self.run["image_preds"].log(self._File.as_image(img))

    def log_text(self, text, timestamp=None, **kwargs):
        log_name = 'text'
        x = None
        y = text
        self.experiment.log_text(log_name, x, y, timestamp)

    def log_dataframe(self, key, dataframe):
        self.run[key].upload(self._File.as_html(dataframe))

    def log_model(self, path, name='model'):
        self.run[name].upload(str(path))

    def save(self):
        pass

    def stop(self):
        self.run.stop()


class WandBLogger(GenericLogger):
    pass
    # todo: add wandblogger


if __name__ == "__main__":
    from PIL import Image

    logger = MultiLogger(
        root=Path("C:/Users/ugur/Documents/GitHub/ai-framework"),
        project_name='machining',
        experiment_name='test2',
        params={
            'param1': 1,
            'param2': 2
        },
        hyperparams={
            'hyperparam1': '1t',
            'hyperparam2': '2t'
        },
    )
    for epoch in range(10):
        logger.log_metric(log_name='training_loss', x=epoch, y=epoch + 1)
        logger.log_metric(log_name='validation_loss', x=epoch, y=epoch + 3)
        logger.log_image(log_name='img',
                         x=epoch,
                         y=Image.open(Path('../../dereotu.jpeg')),
                         image_name=f"image-{epoch}")
        logger.log_text(log_name='text', x=epoch, y=f'{epoch} - Some Text')
        logger.save()

    logger.stop()
