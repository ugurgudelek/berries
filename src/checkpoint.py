import os
import shutil
import time
import torch


class Checkpointer:

    def __init__(self, experiment_dir):
        """
        
        Args:
            experiment_dir: 
        """""

        self.experiment_dir = experiment_dir

    def save(self, epoch, model, optimizer, history):
        """

        Args:
            epoch:
            model:
            optimizer:
            history:

        Returns:

        """

        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        save_path = os.path.join(self.experiment_dir, 'checkpoints', date_time)

        if os.path.exists(save_path):
            # Clear dir
            # os.removedirs(subdir_path) fails if subdir_path is not empty.
            shutil.rmtree(save_path)

        os.makedirs(save_path)

        # SAVE
        torch.save({'epoch': epoch,
                    'optimizer': optimizer,
                    'history': history},
                   os.path.join(save_path, 'optimizer_epoch_history.pt'))

        torch.save(model, os.path.join(save_path, 'model.pt'))

        # with open(os.path.join(subdir_path, self.INPUT_FILE), 'wb') as fout:
        #     dill.dump(self.input, fout)
        # with open(os.path.join(subdir_path, self.OUTPUT_FILE), 'wb') as fout:
        #     dill.dump(self.output, fout)

    @classmethod
    def load(cls, path):
        """

        Args:
            path (str):

        Returns:
            Checkpoint:

        """
        resume_checkpoint = torch.load(os.path.join(path, 'optimizer_epoch_history.pt'))
        model = torch.load(os.path.join(path, 'model.pt'))

        # with open(os.path.join(path, cls.INPUT_FILE), 'rb') as fin:
        #     input = dill.load(fin)
        #
        # with open(os.path.join(path, cls.OUTPUT_FILE), 'rb') as fin:
        #     output = dill.load(fin)

        return (model,
                resume_checkpoint['optimizer'],
                resume_checkpoint['epoch'],
                resume_checkpoint['history'])

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """

        Precondition: Assumes experiment_path exists and have at least 1 checkpoint

        Args:
            experiment_path:

        Returns:

        """
        checkpoints_path = os.path.join(experiment_path, 'checkpoints')
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])
