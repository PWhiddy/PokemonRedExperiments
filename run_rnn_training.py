from torch import nn

from gb_dataset import GBDataset
from rnn_model import RNNModel
from rnn_trainer import RNNTrainer, RNNTrainerConfig

def run_rnn_training(data_path):
    gbd = GBDataset(data_path)
    model = RNNModel()
    trainer = RNNTrainer(model, gbd, RNNTrainerConfig(batch_size=2))
    trainer.train()

if __name__ == '__main__':
    run_rnn_training('rolloutsv1')