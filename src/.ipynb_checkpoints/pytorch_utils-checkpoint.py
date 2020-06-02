import os
import time
import math
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def _naming_scheme(version, epoch, seed, dir=False):
    """a func for converting a comb of version, epoch, seed to a filename with a fixed naming_scheme

    Parameters
    ----------
    version : convertable to str
        The version name of the Checkpoint
    epoch : str or int
        the save type: -1 for last model, an int for a specific epoch, 'best' for best epoch
    seed : int
        random seed used in the model
    dir : bool, optional
        returns the version dir name

    Returns
    -------
    str
        the Checkpoint filename if dir=False
        the version dir name if dir=True
    """
    if dir:
        return '{}'.format(version)
    if isinstance(epoch, int):
        epoch = "{:3d}".format(epoch)
    return os.path.join('{}'.format(version), 'checkpoint_{}_seed-{}_epoch-{}.pth'.format(version, epoch, seed))

def load_model(version=None, models_path=None, epoch=-1, seed=42, prints=True, naming_scheme=_naming_scheme, explicit_file=None):
    """a func for loading a Checkpoint using a comb of version, epoch, seed usind the dill module

    Parameters
    ----------
    version : convertable to str, optional if is given explicit_file
        The version name of the Checkpoint (default is None)
    models_path : str, optional if is given explicit_file
        The full or relative path to the versions dir (default is None)
    epoch : str or int, optional
        the save type: -1 for last model, an int for a specific epoch, 'best' for best epoch (default is -1)
    seed : int, optional
        random seed used in the model (default is 42)
    prints : bool, optional
        if prints=True some training statistics will be printed (default is True)
    naming_scheme : func(version, epoch, seed), optional
        a func that gets version, epoch, seed and returns a str (default is _naming_scheme)
    explicit_file : str, optional
        an explicit path to a Checkpoint file (default is None)

    Returns
    -------
    Checkpoint
        the loaded Checkpoint
    """
    import dill
    
    if from_file is None:
        model_path = os.path.join(models_path, naming_scheme(version, epoch, seed))
    else:
        model_path = explicit_file
    try:
        with open(model_path, "rb") as f:
            model = dill.load(f)

    except Exception as e:
        print("Loading Error")
        raise e

    if prints:
        print("model version:", checkpoint.version)
        print("Number of parameters {} ".format(sum(param.numel() for param in checkpoint.model.parameters())) + 
                         "trainable {}".format(sum(param.numel() for param in checkpoint.model.parameters() if param.requires_grad)))
        if checkpoint.get_log() > 0:
            print("epochs: {}\ntrain_time: {:.3f}\n".format(checkpoint.get_log('epoch'), checkpoint.get_log('train_time')))
            print("last train_loss: {:.5f}".format(checkpoint.get_log('train_loss')))
            print("last val_loss: {:.5f}".format(checkpoint.get_log('val_loss')))
            print("last train_score: {:.5f}".format(checkpoint.get_log('train_score')))
            print("last val_score: {:.5f}".format(checkpoint.get_log('val_score')))
            print("best val_score: {:.5f} at epoch {:d}".format(checkpoint.get_log('val_score'), checkpoint.get_log('epoch', epoch='best')))
        else:
            print("untrained model")
    return checkpoint

# def decision_func(x):
    # return x

class SimpleNet(nn.Module):
    """a simplified nn.Module instance for creating a FCNN model from a list representation of an architecture

    Methods
    -------
    forward
        forward pass
    set_p_dropout
        a method for changing all nn.Dropout modules probability to a gived probability

    Examples
    --------

    >>> model = SimpleNet([512, 256, 'relu', 'dropout', 1, 'sigmoid'])

    >>> model = SimpleNet([512,                # int <input_layer_size>
    >>>                    256, 'relu',        # int <layer_size>, str <activation_function>, activation_function can be 'relu' or 'sigmoid' or 'elu' or 'leaky_relu'
    >>>                    128, 'sigmoid',
    >>>                    'dropout',          # a dropout layer with initial p=0.5 (can be later changed with set_p_dropout method)
    >>>                    64, 'elu',
    >>>                    32, 'leaky_relu',
    >>>                    'dropout',
    >>>                    1, 'sigmoid',
    >>>                   ])
    """

    def __init__(self, architecture):
        """
        Parameters
        -------
        architecture : iterable
            an iterable of a general form:
                * first argument must be an int <input_layer_size>
                * a combination of int <layer_size>, str <activation>, str <'dropout'>
                * int <output_layer_size>, int (optional) <output_layer_activation>

            Example
            -------
            >>> architecture = [512, 256, 'relu', 'dropout', 1, 'sigmoid']

            >>> architecture = [512,                # int <input_layer_size>
            >>>                 256, 'relu',        # int <layer_size>, str <activation_function>, activation_function can be 'relu' or 'sigmoid' or 'elu' or 'leaky_relu'
            >>>                 128, 'sigmoid',
            >>>                 'dropout',          # a dropout layer with initial p=0.5 (can be later changed with set_p_dropout method)
            >>>                 64, 'elu',
            >>>                 32, 'leaky_relu',
            >>>                 'dropout',
            >>>                 1, 'sigmoid',
            >>>                ]
        """
        super(SimpleNet, self).__init__()
        prev_layer = 0
        for i, layer in enumerate(architecture):
            if prev_layer == 0 and isinstance(layer, int):
                prev_layer = layer
                continue
            if isinstance(layer, int):
                exec(f'self.layer_{i} = nn.Linear({prev_layer}, {layer})')
                prev_layer = layer
            if layer == 'dropout':
                exec(f'self.layer_{i} = nn.Dropout(p=0.1)')
            if layer == 'sigmoid':
                exec(f'self.layer_{i} = nn.Sigmoid()')
            if layer == 'relu':
                exec(f'self.layer_{i} = nn.ReLU()')
            if layer == 'elu':
                exec(f'self.layer_{i} = nn.ELU(alpha=1.0)')
            if layer == 'leaky_relu':
                exec(f'self.layer_{i} = nn.LeakyReLU(negative_slope=0.01)')
            if layer == 'prelu':
                exec(f'self.layer_{i} = nn.PReLU(num_parameters=1)')

    def forward(self, x):
        for i in vars(self)['_modules']:
            x = vars(self)['_modules'][i](x)
        return x

    def set_p_dropout(self, p_dropout):
        """
        a method that sets the dropout probabilities of all nn.Dropout layers to a given p_dropout

        Parameters
        -------
        p_dropout : float, 0 < p_dropout < 1
            a probability for all nn.Dropout layers
        """
        for i in vars(self)['_modules']:
            if isinstance(vars(self)['_modules'][i], nn.Dropout):
                vars(self)['_modules'][i] = nn.Dropout(p=p_dropout)


class Checkpoint():
    """
    a wrapper class that manages all interactions with an nn.Module instance,
    including training, saving, evaluating, stats logging,
    setting training hyperparameters, plotting training stats, monitoring training

    Attributes
    -------
    version : convertable to str
        an identifier for the class instance version
    seed : int
        a constant seed to be used in all model interactions
    models_path : str
        path to be used for versions dirs
    model : nn.Module instance
        the model
    optimizer : Object
        the optimizer
    criterion : callable
        the loss function
    naming_scheme : callable
        the naming_scheme of the class instance, see _naming_scheme for an example
    score : callable
        the scoring function to be used for evaluating the train ans validation score,
        can be any function that gets y_true, y_pred args,
        Example : >>> sklearn.metrics.roc_auc_score(y_true, y_pred)
    decision_func : callable
        an optional feature for the case when there needs to be another transformation
        on the raw model output before passing it to the scoring function
        (default is lambda x : x)
    log : pd.DataFrame
        a pd.DataFrame that loggs stats each epoch

    Methods
    -------
    get_log(col='epoch', epoch=-1)
        get a stat from an instance of class
    save(best=False, epoch=False, explicit_file=None)
        save an instance of the class to a dir managed by the class
    plot_checkpoint(attributes, plot_title, y_label, scale='linear', basey=10)
        plots stats of an instance of the class
    train(device, train_dataset, val_dataset, train_epochs=0, batch_size=64,
          optimizer_params={}, prints=True, p_dropout=0, epochs_save=1, lr_decay=0.0, save=False)
        performs a training session on the class instance

    Examples
    --------
    >>> checkpoint = Checkpoint(models_path='models',
    >>>                         version=1.0,
    >>>                         model=model,
    >>>                         score=sklearn.metrics.roc_auc_score,
    >>>                         decision_func=lambda x : x,  # a function that (optionally) converts model output for the score function (default is the identity function)
    >>>                         seed=42,  # int
    >>>                         optimizer=torch.optim.Adam,  # any optimizer
    >>>                         criterion=nn.BCELoss,  # any loss function
    >>>                         naming_scheme=_naming_scheme,
    >>>                         save=False,  # bool
    >>>                         prints=False)  # bool
    """

    def __init__(self, models_path, version, model, score, decision_func=lambda x : x, seed=42,
                 optimizer=torch.optim.Adam, criterion=nn.BCELoss, naming_scheme=_naming_scheme, save=False, prints=False):
        """
        Parameters
        -------
        models_path : str
            path to be used for versions dirs
        version : convertable to str
            an identifier for the class instance version
        model : nn.Module instance
            the model
        score : callable
            the scoring function to be used for evaluating the train ans validation score,
            can be any function that gets y_true, y_pred args,
            Example : >>> sklearn.metrics.roc_auc_score(y_true, y_pred)
        decision_func : callable, optional
            an optional feature for the case when there needs to be another transformation
            on the raw model output before passing it to the scoring function
            (default is lambda x : x)
        seed : int, optional
            a constant seed to be used in all model interactions (default is 42)
        optimizer : Object, optional
            the optimizer (default is torch.optim.Adam)
        criterion : callable, optional
            the loss function (default is nn.BCELoss)
        naming_scheme : callable, optional
            the naming_scheme of the class instance (default is _naming_scheme)
        save : bool, optional
            if save=True, saves the class instance (default is False)
        prints : bool, optional
            if prints=True, prints the model version and num of model parameters and num of model trainable parameters (default is False)

        Examples
        --------
        >>> checkpoint = Checkpoint(models_path='models',
        >>>                         version=1.0,
        >>>                         model=model,
        >>>                         score=sklearn.metrics.roc_auc_score,
        >>>                         decision_func=lambda x : x,  # a function that (optionally) converts model output for the score function (default is the identity function)
        >>>                         seed=42,  # int
        >>>                         optimizer=torch.optim.Adam,  # any optimizer
        >>>                         criterion=nn.BCELoss,  # any loss function
        >>>                         naming_scheme=_naming_scheme,
        >>>                         save=False,  # bool
        >>>                         prints=False)  # bool
        """
        self.version = version
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.models_path = models_path
        self.naming_scheme = naming_scheme
        self.model = model
        self.optimizer = optimizer([p for p in self.model.parameters() if p.requires_grad], lr=4e-4)
        self.criterion = criterion()
        self.score = score
        self.decision_func = decision_func
        self.log = pd.DataFrame(columns=['train_time',
                                         'timestamp',
                                         'train_loss',
                                         'val_loss',
                                         'train_score',
                                         'val_score',
                                         'batch_size',
                                         'best'
                                        ] + sorted([param for param in self.optimizer.param_groups[0].keys() if param != 'params']))

        if save:
            self.save()
        if prints:
            print("model version:", self.version)
            print("Number of parameters {} ".format(sum(param.numel() for param in self.model.parameters())) + 
                             "trainable {}".format(sum(param.numel() for param in self.model.parameters() if param.requires_grad)))

    def get_log(self, col='epoch', epoch=-1):
        """
        extracts stats from self.log

        Parameters
        -------
        col : str, optional
            path to be used for versions dirs (default is 'epoch')
        epoch : int or str, optional
            the epoch to load, can be int > 0 or -1 or 'best'  (default is -1)

        Returns
        -------
        str or int or float or None
            returns the log stat in the index (line) epoch
            (if epoch epoch == -1 returns last line,
            if epoch epoch == 'best' returns last index (line) where self.log['best'] == True)
            returns None if epoch index or col column does not exist in the log
        """
        if len(self.log) == 0:
            if col == 'val_loss' or col == 'train_loss':
                return 99999.99999
            else:
                return 0

        if epoch == -1:
            index = self.log.tail(1).index[0]
        elif epoch == 'best':
            try:
                index = self.log[self.log['best'] == True].tail(1).index[0]
            except Exception:
                index = self.log.tail(1).index[0]
        elif isinstance(epoch, int):
            index = epoch

        if col == 'epoch':
            return index
        else:
            try:
                return self.log[col].loc[index]
            except Exception:
                return None

    def _get_optimizer_params(self):
        param_list = sorted([param for param in self.optimizer.param_groups[0].keys() if param != 'params'])
        return OrderedDict({param: self.optimizer.param_groups[0][param] for param in param_list})

    def save(self, best=False, epoch=False, explicit_file=None):
        """
        saves the class instance using self.naming_scheme

        Parameters
        -------
        best : bool, optional
            aditionally saves the model to a 'best' epoch file (default is False)
            Example file name : 'checkpoint_1_seed-42_epoch-best.pth'
        epoch : bool, optional
            aditionally saves the model to an epoch file (default is False)
            Example file name : 'checkpoint_1_seed-42_epoch-001.pth'
        explicit_file : str, optional
            if explicit_file is not None, saves the model to an explicitly specified explicit_file name (default is None)
        """
        import dill

        if explicit_file is not None:
            with open(explicit_file, 'wb') as f:
                dill.dump(self, f)
            return

        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)
        if not os.path.exists(os.path.join(self.models_path, self.naming_scheme(self.version, -1, self.seed, dir=True))):
            os.mkdir(os.path.join(self.models_path, self.naming_scheme(self.version, -1, self.seed, dir=True)))

        with open(os.path.join(self.models_path, self.naming_scheme(self.version, -1, self.seed)), 'wb') as f:
            dill.dump(self, f)
        if best:
            with open(os.path.join(self.models_path, self.naming_scheme(self.version, 'best', self.seed)), 'wb') as f:
                dill.dump(self, f)
        if epoch:
            with open(os.path.join(self.models_path, self.naming_scheme(self.version, self.get_log(), self.seed)), 'wb') as f:
                dill.dump(self, f)

    def plot_checkpoint(self, attributes, plot_title, y_label, scale='linear', basey=10, save=False):
        """
        plots stats of the class instance

        Parameters
        ----------
        attributes : iterable of str
            an iterable of self.log.columns to plot
        plot_title : str
            the plot title to display
        y_label : str
            the y label to display
        scale : str, optional
            plot scale, if scale='log' plots y axis in log scale,
            otherwise plots y axis in linear scale (default is 'linear')
        basey : int, optional
            used if scale='log', log base (default is 10)
        save : bool, optional
            saves plot to <plot_title>.png in version dir (default is False)
        """
        if not self.get_log():
            print("model have not trained yet")
            return
        epochs = self.log.index
        to_plot = []
        for attribute in attributes:
            to_plot.append(self.log[attribute])
        min_e = np.min(epochs)
        max_e = np.max(epochs)
        for data in to_plot:
            plt.plot(epochs, data)
        plt.xlim(min_e - (max_e - min_e)*0.02, max_e + (max_e - min_e)*0.02)
        plt.xlabel('epoch')
        plt.ylabel(y_label)
        if scale == 'log':
            plt.yscale(scale, basey=basey)
        else:
            plt.yscale(scale)
        plt.legend(attributes)
        plt.title(plot_title)
        if save:
            plt.savefig(os.path.join(self.models_path, self.naming_scheme(self.version, -1, self.seed, dir=True), '{}.png'.format(plot_title)), dpi=200)
        plt.show()

    def _run(self, device, data_loader, train=False):
        """
        a private method used to pass data through model
        if train=True : computes gradients and updates model weights
        if train=False : returns loss and score
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if train:
            self.model.train()
        else:
            self.model.eval()
            loss_sum = np.array([])
            y_pred = np.array([])
            y_true = np.array([])

        for batch in data_loader:
            for i, _ in enumerate(batch):
                batch[i] = batch[i].to(device)
            out = self.model(*batch[:-1])
            loss = self.criterion(out, batch[-1].float())
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                loss_sum = np.append(loss_sum, float(loss.data))
                y_pred = np.append(y_pred, self.decision_func(out.detach().cpu().numpy()))
                y_true = np.append(y_true, batch[-1].detach().cpu().numpy())

        if not train:
            return float(loss_sum.mean()), float(self.score(y_true, y_pred))

    def train(self, device, train_dataset, val_dataset, train_epochs=0, batch_size=64, optimizer_params={}, prints=True, p_dropout=0, epochs_save=0, lr_decay=0.0, save=False):
        """
        performs a training session

        Parameters
        ----------
        device : str or torch.device
            device to be used for training,
            Example : >>> torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataset : torch.utils.data.TensorDataset
            train dataset
        val_dataset : torch.utils.data.TensorDataset
            validation dataset
        train_epochs : int, optional
            number of epochs to train (default is 0)
        batch_size : int, optional
            batch size for the training session (default is 64)
        optimizer_params : dict, optional
            optimizer params for the training session (default is {})
        prints : bool, optional
            if prints=True, prints stats after each epoch(default is True)
        p_dropout : float, optional
            for the training session (default is 0)
        epochs_save : int, optional
            if save=True and epochs_save>0, saves model each epochs_save epochs to an epoch file (default is 0)
        lr_decay : float, optional
            decreases optimizer learning rate by lr_decay percentage each epoch (default is 0.0)
        save : bool, optional
            if save=True, saves the model after each epoch, also saves best model when new best is found
            if epochs_save>0, also saves saves model each epochs_save epochs (default is False)
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if train_epochs > 0:
            self.model.to(device)
            start_epoch = self.get_log()
            start_time = self.get_log('train_time')

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

            for param in optimizer_params:
                for group, _ in enumerate(self.optimizer.param_groups):
                    self.optimizer.param_groups[group][param] = optimizer_params[param]
            if callable(getattr(self.model, "set_p_dropout", None)):
                self.model.set_p_dropout(p_dropout)
            params = self._get_optimizer_params()

            # train for num_epochs epochs
            tic = time.time()

            for train_epoch in range(train_epochs):
                epoch = train_epoch + start_epoch + 1
                # train epoch
                self._run(device, train_loader, train=True)
                with torch.no_grad():
                    train_loss, train_score = self._run(device, train_loader, train=False)
                    val_loss, val_score = self._run(device, val_loader, train=False)

                # save sample to checkpoint
                best = val_score > self.get_log('val_score', epoch='best')
                train_time = float(start_time + (time.time() - tic)/60)

                to_log = [train_time,
                          time.strftime('%H:%M:%S %d-%m-%Y'),
                          train_loss,
                          val_loss,
                          train_score,
                          val_score,
                          batch_size,
                          best,
                          ] + list(self._get_optimizer_params().values())
                self.log.loc[epoch] = to_log

                # epoch progress prints
                if prints:
                    print('epoch {:3d}/{:3d} | train_loss {:.5f} | val_loss {:.5f} | train_score {:.5f} | val_score {:.5f} | train_time {:6.2f} min'
                          .format(epoch, train_epochs + start_epoch, train_loss, val_loss, train_score, val_score, train_time))

                # save checkpoint
                if save and epochs_save > 0:
                    self.save()
                    if epoch % epochs_save == 0:
                        self.save(epoch=True)
                    if best:
                        self.save(best=True)

                # lr_decay
                for group, _ in enumerate(self.optimizer.param_groups):
                    self.optimizer.param_groups[group]['lr'] = self.optimizer.param_groups[group]['lr']*(1 - lr_decay)

            # # end of session prints
            # if prints:
                # print("train_loss {:.6f} | prev {}\nval_loss {:.6f} | prev {}"
                      # .format(self.get_log('train_loss'), self.get_log('train_loss', start_epoch),
                             # self.get_log('val_loss'), self.get_log('val_loss', start_epoch)))
                # print("train_score {:.6f} | prev {}\nval_score {:.6f} | prev {}"
                      # .format(self.get_log('train_score'), self.get_log('train_score', start_epoch),
                             # self.get_log('val_score'), self.get_log('val_score', start_epoch)))


