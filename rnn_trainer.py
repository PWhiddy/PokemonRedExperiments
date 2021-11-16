
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

class RNNTrainerConfig:
    # optimization parameters
    max_epochs = 4
    batch_size = 16
    learning_rate = 3e-4
    #betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    num_workers = 4 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class RNNTrainer:

    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def train(self):

        # TODO separate weight decay parameters from rest
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.max_epochs):

            loader = DataLoader(
                self.dataset, shuffle=False, pin_memory=False,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (f, a, la, r) in pbar:
                frames, actions, last_actions, rewards_to_go = f.to(self.device), a.to(self.device), la.to(self.device), r.to(self.device)
                #print(f'frame: {frames.shape} actions: {actions.shape} last_actions: {last_actions.shape} rewards_to_go: {rewards_to_go.shape}')
                self.model.zero_grad()
                action_logits, loss = self.model(frames, actions, last_actions, rewards_to_go)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                optimizer.step()
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")
