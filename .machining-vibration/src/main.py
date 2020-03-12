__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import random
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pytorch imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# scikit-learn imports
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix

# local imports
from dataset import VibrationDataset

from model import CNN2D



if __name__ == "__main__":

    EXPERIMENT_NAME = 'bati-cnn'
    seed = 7
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_epochs = 1000
    batch_size = 2
    learning_rate = 1e-4
    train_ratio = 0.75

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)

    # dataset = MNIST('../data/MNIST', train=True, transform=img_transform, download=True)
    # train_dataloader = DataLoader(dataset = MNIST('../data/MNIST', train=True, transform=img_transform, download=True), batch_size=batch_size)
    # test_dataloader = DataLoader(dataset = MNIST('../data/MNIST', train=False, transform=img_transform, download=True), batch_size=batch_size)
    dataset = VibrationDataset(path=Path('D:/machining/chatter_data/preprocessed/alu_v1'),
                               train_ratio=train_ratio, kind='acc', shuffle_mode=1)
    # dataset = TimeSeriesDataset(train_ratio=train_ratio)

    # TimeSeriesDataset.feature_heatmap(dataset.FEATURES, dataset.train_dataset)
    # TimeSeriesDataset.feature_importance(dataset.FEATURES, dataset.train_dataset, dataset.test_dataset)

    train_dataloader = DataLoader(dataset.train_dataset, batch_size=batch_size,
                                  # sampler=ImbalancedDatasetSampler(dataset.train_dataset)
                                  )
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=1)

    # model = CNN1D().to(device)
    model = CNN2D().to(device)
    # model = LogisticRegression(input_dim=dataset.dataset.features[0].shape[0],
    #            output_dim=1).to(device)
    # model = LogisticRegression(input_dim=dataset.dataset.features[0].shape[0]).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.01, nesterov=True)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//5, gamma=0.1)

    # net = skorch.NeuralNetClassifier(model, max_epochs=num_epochs,
    #                            criterion=nn.CrossEntropyLoss,
    #                            optimizer=torch.optim.Adam,
    #                                  optimizer__lr=learning_rate,
    #                                  iterator_train__batch_size=batch_size,
    #                                  iterator_train__sampler=ImbalancedDatasetSampler(dataset.train_dataset))
    # net.fit(dataset.dataset.features.astype('float32'),
    #         dataset.dataset.labels.astype('int64'))
    #
    #
    # print("Test Score:", net.score(dataset.test_dataset.features.astype('float32'),
    #                               dataset.test_dataset.labels.astype('int64')))

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train_accuracies_bucket = Bucket()
    # test_accuracies_bucket = Bucket()
    train_losses = []
    test_losses = []
    # train_accuracies = []
    # test_accuracies = []
    for epoch in range(num_epochs):

        # accuracies = []
        losses = []
        # predictions = []
        # proba0s, proba1s = [], []
        labels = []
        # print(f"lr:{optimizer.param_groups[0]['lr']}")
        # scheduler.step()
        for data in train_dataloader:
            img, label = data[0].to(device).float(), data[1].to(device).long() # classification
            # img, label = data[0].to(device).float(), data[1].to(device).float()  # regression

            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracies.append(torch.mean((torch.argmax(output, dim=1) == label).float()).item())
            losses.append(loss.item())
            # predictions += list((torch.argmax(output, dim=1).numpy()))
            labels += list(label.numpy())
            # proba = torch.nn.functional.softmax(output.detach(), dim=1).numpy()
            # proba0s += list(proba[:, 0])
            # proba1s += list(proba[:, 1])

        train_losses.append(np.array(losses).mean())
        # train_accuracies.append(np.array(accuracies).mean())
        # train_accuracies_bucket.put(train_accuracies[-1])
        print(f'[Train]  epoch [{epoch + 1}/{num_epochs}], '
              f'loss:{train_losses[-1]:.4f}, '
              # f'acc:{train_accuracies[-1]:.4f}, '
              # f'predict_mean:{np.array(predictions).mean():.4f}, '
              f'label_mean:{np.array(labels).mean():.4f}, '
              # f'acc_mean10:{train_accuracies_bucket.mean():.4f}',
              )

        train_pred_df = pd.DataFrame({'label': labels,
                                      # 'pred': predictions,
                                      # 'proba0': proba0s,
                                      # 'proba1': proba1s,
                                      })

        # accuracies = []
        losses = []
        # predictions = []
        # proba0s, proba1s = [], []
        labels = []
        model.train(mode=False)
        for data in test_dataloader:
            img, label = data[0].to(device).float(), data[1].to(device).long()
            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad()

            # accuracies.append(torch.mean((torch.argmax(output, dim=1) == label).float()).item())
            losses.append(loss.item())
            # predictions += list((torch.argmax(output, dim=1).numpy()))
            labels += list(label.numpy())
            # proba = torch.nn.functional.softmax(output.detach(), dim=1).numpy()
            # proba0s += list(proba[:, 0])
            # proba1s += list(proba[:, 1])

        test_losses.append(np.array(losses).mean())
        # test_accuracies.append(np.array(accuracies).mean())
        # test_accuracies_bucket.put(test_accuracies[-1])
        print(f'[Test ]  epoch [{epoch + 1}/{num_epochs}], '
              f'loss:{test_losses[-1]:.4f}, '
              # f'acc:{test_accuracies[-1]:.4f}, '
              # f'predict_mean:{np.array(predictions).mean():.4f}, '
              f'label_mean:{np.array(labels).mean():.4f}, '
              # f'acc_mean10:{test_accuracies_bucket.mean():.4f}'
              )
        model.train(mode=True)
        test_pred_df = pd.DataFrame({'label': labels,
                                     # 'pred': predictions,
                                     # 'proba0': proba0s,
                                     # 'proba1': proba1s,
                                     })

        if (epoch + 1) % 50 == 0:
            output_dir = f'../results/{EXPERIMENT_NAME}/{epoch}'
            os.makedirs(output_dir, exist_ok=True)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            # plt.clf()

            sns.lineplot(range(train_losses.__len__()), train_losses, label='training loss')
            sns.lineplot(range(train_losses.__len__()), test_losses, label='test loss')
            # sns.lineplot(range(train_losses.__len__()), train_accuracies, label='training accuracy')
            # sns.lineplot(range(train_losses.__len__()), test_accuracies, label='test accuracy')
            ax.set_xlabel("Epoch")
            # plt.show(block=False)
            # plt.pause(0.001)
            plt.ylim((0., 1.5))
            plt.savefig(os.path.join(output_dir, 'lr_curve.png'))


            # cm_train = confusion_matrix(train_pred_df['label'], train_pred_df['pred'])
            # cm_test = confusion_matrix(test_pred_df['label'], test_pred_df['pred'])
            # utils.plot_confusion_matrix(y_true=train_pred_df['label'],
            #                             y_pred=train_pred_df['pred'],
            #                             classes=['no chatter', 'medium chatter', 'high chatter'],
            #                             save_path=os.path.join(output_dir, 'cm_train.png'))
            # utils.plot_confusion_matrix(y_true=test_pred_df['label'],
            #                             y_pred=test_pred_df['pred'],
            #                             classes=['no chatter', 'medium chatter', 'high chatter'],
            #                             save_path=os.path.join(output_dir, 'cm_test.png'))

            train_pred_df.to_csv(os.path.join(output_dir, 'train_pred.csv'))
            test_pred_df.to_csv(os.path.join(output_dir, 'test_pred.csv'))
            # np.savetxt(os.path.join(output_dir, 'cm_train.txt'), cm_train)
            # np.savetxt(os.path.join(output_dir, 'cm_test.txt'), cm_test)
            pd.DataFrame({'train_loss': train_losses,
                          'test_loss': test_losses,
                          # 'train_acc': train_accuracies,
                          # 'test_acc': test_accuracies,
                          }).to_csv(os.path.join(output_dir, 'result.csv'))


    torch.save(model.state_dict(), os.path.join(output_dir, '..', 'vibration_model.pth'))
