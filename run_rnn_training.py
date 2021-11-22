
from gb_dataset import GBDataset
from rnn_model import RNNModel
from rnn_trainer import RNNTrainer, RNNTrainerConfig

def run_rnn_training(data_path):
    gbd = GBDataset(data_path, limit_steps=20)
    model = RNNModel(vis_emb_dim=32, hidden_dim=16)
    trainer = RNNTrainer(model, gbd, RNNTrainerConfig(batch_size=16, learning_rate=3e-3))
    trainer.train()

if __name__ == '__main__':
    run_rnn_training('rolloutsv1')