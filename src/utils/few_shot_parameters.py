from torch import optim

from prototypical_loss import PrototypicalLoss
from utils.dataloader import (load_meta_test_dataloader, load_meta_train_dataloaders)


class FewShotParameters(object):

    @staticmethod
    def get_params(dataset_list, model=None):
        if model:
            return TrainingParameters(model, dataset_list)
        else:
            return TestParameters(dataset_list)

class TrainingParameters(FewShotParameters):

    def __init__(self, train_val_sets, model):
        # Constants
        LEARNING_RATE = 1e-3
        N_SUPPORT = 5
        N_QUERY = 15
        SAMPLES_PER_CLASS = N_SUPPORT + N_QUERY        
        N_WAYS = (5, 5)

        # Variables
        self.patience_limit = 20
        self.n_episodes = 100
        self.n_epochs = 10000
        self.l1_lambda = 0.1


        self.train_loader, self.valid_loader = load_meta_train_dataloaders(train_val_sets,
                                                                           samples_per_class=SAMPLES_PER_CLASS,
                                                                           n_episodes=self.n_episodes,
                                                                           classes_per_it=N_WAYS)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=LEARNING_RATE)

        self.criterion = PrototypicalLoss(N_SUPPORT)

        # Reduce the learning rate by half every 2000 episodes
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=20,  # 20 epochs of 100 episodes
                                                   gamma=0.5)


class TestParameters(FewShotParameters):

    def __init__(self, test_set):
        # Constants
        N_SUPPORT = 5
        N_QUERY = 15
        N_WAY = 5
        SAMPLES_PER_CLASS = N_SUPPORT + N_QUERY

        self.criterion = PrototypicalLoss(N_SUPPORT)
        self.n_episodes = 100
        self.n_epochs = 6

        self.test_loader = load_meta_test_dataloader(test_set,
                                                     samples_per_class=SAMPLES_PER_CLASS,
                                                     n_episodes=self.n_episodes,
                                                     classes_per_it=N_WAY)
