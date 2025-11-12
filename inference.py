import torch
import arabic_reshaper
from bidi.algorithm import get_display
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

# ----------------- Load Dataset -------------------
with open('dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
checkpoint = torch.load(checkpoint_path, map_location=device)
model = GPTLanguageModel(vocab_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load vocabulary
itos = checkpoint['itos']
stoi = checkpoint['stoi']
decode = lambda l: ''.join([itos[i] for i in l])
encode = lambda s: [stoi[c] for c in s]

# ----------------- Generation Loop -------------------
print("Starting continuous text generation. Press Ctrl+C to stop.\n")

output_file = "generated_text.txt"
max_tokens_per_save = 1000  # save every 10,000 tokens
generated_tokens = []

# start from a custom sentence
start_sentence = "هنا البداية"
context = torch.tensor([encode(start_sentence)], dtype=torch.long, device=device)
generated_tokens.extend(context[0].tolist())  # store initial tokens

total_generated = 0

try:
    while True:
        max_new_tokens = 100
        with torch.no_grad():
            idx = model.generate(context, max_new_tokens=max_new_tokens)
        
        new_tokens = idx[0, context.shape[1]:].tolist()
        generated_tokens.extend(new_tokens)
        total_generated += len(new_tokens)

        generated_text = decode(new_tokens)
        reshaped_text = arabic_reshaper.reshape(generated_text)
        bidi_text = get_display(reshaped_text)
        print(bidi_text, end='', flush=True)

        context = idx[:, -block_size:]

        if total_generated >= max_tokens_per_save:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(decode(generated_tokens))
                f.write("\n\n--- Saved checkpoint ---\n\n")
            print(f"\n\nSaved {total_generated} tokens to {output_file}")
            generated_tokens = []
            total_generated = 0

except KeyboardInterrupt:
    if generated_tokens:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(decode(generated_tokens))
            f.write("\n\n--- Saved on exit ---\n\n")
    print("\n\nGeneration stopped by user and text saved.")
