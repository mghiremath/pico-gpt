import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6  # 384/6 = 64 -every head's dimsension
n_layer = 6
dropout = 0.2
# ----------------------------------------------------------------------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) # the size of the vocabulary

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# we will use 90% of the data for training and 10% for validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # we don't need gradients for this 
def estimate_loss():
    out = {}
    model.eval() # set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set the model back to training mode
    return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # Projection is just the linear transformation
        nn.Dropout(dropout), # dropout - we can use dropout to prevent overfitting, right before the final layer and before the residual connections back into the pathway
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)  # Projection is just the linear transformation of this output
        return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #
        self.droupout = nn.Dropout(dropout) # dropout - we can use dropout to prevent overfitting, right before the final layer and before the residual connections back into the pathway
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x)  # (B, T, C) These are learnable projections to transform input into keys, queries, and values for attention.
        # compue attendtion scores ("affinities")
        weights = q @ k.transpose(-2,-1) # (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T]==0, float('-inf')) # for all tril elements that , make them -inf
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.droupout(weights)
        # perform weighted aggregation of values
        out = weights @ v # (B, T, T) @  (B, T, C) --->  (B, T, C)

        return out

class FeedForward(nn.Module):
    """" a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # 4*n_embd is the number of hidden units(referring the paper)
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # Projection is just the linear transformation
            nn.Dropout(dropout), # dropout - we can use dropout to prevent overfitting, right before the final layer and before the residual connections back into the pathway
        )
        
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation. This disperses the communication and computation of a transformer """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # disperses communication - # multi head self-attention - 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd) # disperses computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # we can use residual connections to fork off the communication and come back 
        x = x + self.ffwd(self.ln2(x)) #fork the computation and come back
        return x
    
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #to embed each token as a vector of size n_embd.
        # we want to encode the token position along with their identities
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # to embed each position as a vector of size n_embd. it is an embedding of block size by n_embd - each position from 0->block_size-1 will also gets their own embedding vector
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size) # language model head initialising a linear layer

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embd = self.token_embedding_table(idx) # (B,T,C) # we are encoding the identity of the tokens by taking in indices
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
        x = token_embd + pos_embd #(B, T, C)
        x = self.blocks(x) # apply one head of self-attention. (B, T, C)
        logits = self.lm_head(x) #(B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens else it's going to run out of scope
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))