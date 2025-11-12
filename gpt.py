import torch
from tqdm import trange
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from model import GPTLanguageModel

# ----------------- Hyperparameters -----------------
batch_size = 128        # sequences processed in parallel
block_size = 256        # context length
checkpoint_path = "gpt_model_checkpoint.pth"
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50          # reduced for faster evaluation
n_embd = 512
n_head = 8
n_layer = 6
dropout = 0.2
# ---------------------------------------------------

print(f"Device: {device}")
torch.manual_seed(1337)

# ----------------- Load Dataset -------------------
with open('dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Text Length: ", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# ----------------- Data Loader -------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # pin memory and move to device
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

# ----------------- Evaluation -------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ----------------- Instantiate Model -------------------
model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler()

print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# ----------------- Training Loop -------------------
train_losses = []
val_losses = []

for iter in trange(max_iters):
    # evaluate loss
    if (iter+1) % eval_interval == 0:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': iter,
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi
        }, checkpoint_path)

    xb, yb = get_batch('train')

    optimizer.zero_grad(set_to_none=True)
    with autocast():
        _, loss = model(xb, yb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# ----------------- Plot Losses -------------------
plt.figure(figsize=(8,5))
plt.plot(range(0, max_iters, eval_interval), train_losses, label='Train Loss')
plt.plot(range(0, max_iters, eval_interval), val_losses, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# ----------------- Generate Text -------------------
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))