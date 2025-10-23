import torch
import torch.nn as nn
from torch.nn import functional as F

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
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
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

"""convert dataset to tokens"""
data = torch.tensor(encode(text), dtype=torch.long)
"""train/test split"""
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

"""creating batches of chunks"""


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # sets model to  eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # sets model to train mode
    return out


"""self-attention head"""
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # head size = key space dimension
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  # C = n_embeddings

        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)

        # compute the attention pattern
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)

        # perform weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B,T,C)
        return out

"""Multi-head attention"""
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

"""Feedforward layer"""
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
        )

    def forward(self, x):
        return self.net(x)


"""Transformer block"""
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x



"""Create bigram model"""
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # embed the meaning and the position
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # transformer blocks
        self.blocks = nn.Sequential(
            Block(n_embd, n_heads=4),
            Block(n_embd, n_heads=4),
            Block(n_embd, n_heads=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # C = n_embd
        tok_embd = self.token_embedding_table(idx)  # (batch, time, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_embd + pos_embd  # (B, T, C)
        x = self.blocks(x)  # (B,T,C)
        logits = self.lm_head(x)  # (batch, time, vocab_size)
        if targets is None:
            loss = None
        else:
            # pytorch wants (b, t, c)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop the idx upto block size
            # -block because we want the last part of the input text(including generated stuff in last iteration) and
            # the size of this would be block_size because of positional embedding table
            idx_cond = idx[:, -block_size:]
            # get the prediction
            logits, loss = self(idx_cond)
            # focus on last time step
            logits = logits[:, -1, :]  # logits becomes (B, C) from (B, T, C)
            # apply softmax for getting probabs
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
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

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

"""inferencing on the trained model"""
context = idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(
    decode(
        m.generate(
            context,
            max_new_tokens=500
        )[0].tolist()
    )
)
