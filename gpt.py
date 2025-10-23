import torch
import torch.nn as nn
from torch.nn import functional as F

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------------------------

torch.manual_seed(1337)


"""import from the loaded txt file"""

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()


"""info about dataset"""
chars = sorted(list(set(text)))
vocab_size = len(chars)

"""encoder/decoder (tokenizer)"""
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

"""convert dataset to tokens"""
data = torch.tensor(encode(text), dtype=torch.long)
"""train/test split"""
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


"""creating batches of chunks"""
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval() # sets model to  eval mode
  for split in ['train', 'val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
          X, Y = get_batch(split)
          logits, loss = model(X, Y)
          losses[k] = loss.item()
      out[split] = losses.mean()
  model.train() # sets model to train mode
  return out

"""Create bigram model"""
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    # embed the meaning
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    #embed the position
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size) #

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_embd = self.token_embedding_table(idx) # (batch, time, channel)
    pos_embd = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_embd + pos_embd # (B, T, C)
    logits = self.lm_head(x) # (batch, time, vocab_size)
    if targets is None:
      loss = None
    else:
    # pytorch wants (b, t, c)
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      # get the prediction
      logits, loss = self(idx)
      #focus on last time step
      logits = logits[:, -1, :] # logits becomes (B, C) from (B, T, C)
      # apply softmax for getting probabs
      probs = F.softmax(logits, dim=-1) # (B, C)

      #sample from distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

# running model and getting the logits and loss
model = BigramLanguageModel()
m = model.to(device)

"""# Training the model"""
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter: {iter}: train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


"""inferencing on the trained model"""
context = idx=torch.zeros((1,1), dtype=torch.long, device=device)
print(
    decode(
        m.generate(
            context,
            max_new_tokens=500
        )[0].tolist()
    )
)