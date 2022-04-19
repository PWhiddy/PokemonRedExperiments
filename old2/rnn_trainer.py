import uuid
from pathlib import Path
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
    parallel_scheme = torch.nn.DataParallel
    checkpoint_dir = Path('checkpoints')
    checkpoint_interval = 32

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class RNNTrainer:

    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.instance_id = str(uuid.uuid4())[:8]
        self.config.checkpoint_dir.mkdir(exist_ok=True)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = config.parallel_scheme(self.model).to(self.device)

    def train(self):

        # TODO separate weight decay parameters from rest
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.max_epochs):

            loader = DataLoader(
                self.dataset, shuffle=False, pin_memory=False,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers)
            
            total_acc = 0
            total_loss = 0

            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (f, r, pa, ta) in pbar:
                frames, rewards_to_go, previous_actions, target_actions = [x.to(self.device) for x in [f, r, pa, ta]]
                action_logits, loss, accuracy = self.model(frames, rewards_to_go, previous_actions, target_actions)
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                optimizer.step()
                total_loss += loss.item()
                total_acc += accuracy
                pbar.set_description(
                    f'epoch {epoch+1} iter {it} ' 
                    f'train loss {loss.item():.5f} avg_loss: {total_loss/(it+1):.5f} '
                    f'accuracy: {accuracy:.5f} avg_acc: {total_acc/(it+1):.5f}'
                )
                if it % self.config.checkpoint_interval == 0:
                    checkpoint_name = f'{self.instance_id}_e{epoch}_b{it}.pt'
                    pbar.set_description(f'saving: {checkpoint_name}')
                    torch.save(
                        self.model.state_dict(), 
                        self.config.checkpoint_dir / Path(checkpoint_name)
                    )
