from torch import nn
from torch.nn import functional as F
from einops import rearrange

class RNNModel(nn.Module):

    def __init__(self, vis_emb_dim=256, hidden_dim=128, possible_actions=41):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=3), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=4), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9152, vis_emb_dim),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(vis_emb_dim, hidden_dim, batch_first=True)
        self.action_embed = nn.Sequential(nn.Embedding(possible_actions, vis_emb_dim), nn.Tanh())
        self.reward_to_go_embed = nn.Sequential(nn.Linear(1, vis_emb_dim), nn.Tanh())
        self.action_head = nn.Linear(hidden_dim, possible_actions, bias=False)

    def forward(self, frames, actions, last_actions, reward_to_go):
        # frames in shape [b, 2000, 144, 160, 3]
        batch_size = frames.shape[0]
        conv_input = rearrange(frames, 'b f h w c -> (b f) c h w') 
        conv_out = self.conv_layers(conv_input) # [bxf, vis_emb_dim]
        frame_embeddings = rearrange(conv_out, '(b f) e -> b f e', b=batch_size)
        a_emb = self.action_embed(last_actions)
        r_emb = self.reward_to_go_embed(rearrange(reward_to_go, 'b r -> b r ()'))
        lstm_input = frame_embeddings + a_emb + r_emb
        lstm_out, (hidden_state, hidden_cell) = self.lstm(lstm_input)
        action_logits = self.action_head(lstm_out)
        #print(f'action_logits: {action_logits.shape}')
        #print(f'lstm_out: {lstm_out.shape} hidden_state: {hidden_state.shape} hidden_cell: {hidden_cell.shape}')
        loss = F.cross_entropy(rearrange(action_logits, 'b s e -> (b s) e'), rearrange(actions, 'b s -> (b s)'))
        #print(f'loss: {loss}')
        return action_logits, loss
