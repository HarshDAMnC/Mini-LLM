import torch
import torch.nn.functional as F
import re

#load texxt from file
with open("alice.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

text = re.sub(r"[^a-z\s]", "", text)
words = text.split()
#split text to words and make a vocabuary from it
vocab = sorted(set(words))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

data = torch.tensor([stoi[w] for w in words], dtype=torch.long)

#hyperparams
vocab_size = len(vocab)
dim = 64
seq_len = 8
epochs = 5000
lr = 0.2

token_emb = (torch.randn(vocab_size, dim) * 0.02).requires_grad_()
pos_emb   = (torch.randn(seq_len, dim) * 0.02).requires_grad_()

wq = (torch.randn(dim, dim) * 0.02).requires_grad_()
wk = (torch.randn(dim, dim) * 0.02).requires_grad_()
wv = (torch.randn(dim, dim) * 0.02).requires_grad_()

hidden_dim = 4 * dim
w1 = (torch.randn(dim, hidden_dim) * 0.02).requires_grad_()
b1 = torch.zeros(hidden_dim, requires_grad=True)

w2 = (torch.randn(hidden_dim, dim) * 0.02).requires_grad_()
b2 = torch.zeros(dim, requires_grad=True)

wo = (torch.randn(dim, vocab_size) * 0.02).requires_grad_()

params = [token_emb, pos_emb, wq, wk, wv, w1, b1, w2, b2, wo]

for step in range(epochs):
    i = torch.randint(0, len(data) - seq_len - 1, (1,)).item()
    #input window and desired output window
    x = data[i : i + seq_len]
    y = data[i + 1 : i + 1 + seq_len]
    #conver text to vectors of size(seq_len,dim) and add positional information
    x_emb = token_emb[x] + pos_emb
    #attention layer
    Q = x_emb @ wq
    K = x_emb @ wk
    V = x_emb @ wv

    scores = Q @ K.T / (dim ** 0.5)

    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(mask == 0, -1e9)
    #probability of each token
    attn = F.softmax(scores, dim=-1)
    context = attn @ V

    #FFN
    ff = torch.relu(context @ w1 + b1)
    context = ff @ w2 + b2

    #calculate 
    logits = context @ wo
    loss = F.cross_entropy(logits.view(-1, vocab_size), y)
    
    if torch.isnan(loss):
        print("NaN detected, stopping")
        break
    #backpropogation
    loss.backward()

    with torch.no_grad():
        for p in params:
            p.grad.clamp_(-1.0, 1.0)
            p -= lr * p.grad
            p.grad.zero_()

    if step % 500 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")


start_words = ["alice", "was", "beginning", "to", "get", "very", "tired", "of"]
context = torch.tensor([stoi[w] for w in start_words], dtype=torch.long)

print("\nGenerated text:\n")
#text generation
for _ in range(50):
    x_emb = token_emb[context] + pos_emb
    #take a window of texts and apply the same to it
    Q = x_emb @ wq
    K = x_emb @ wk
    V = x_emb @ wv

    scores = Q @ K.T / (dim ** 0.5)
    scores = scores.masked_fill(mask == 0, -1e9)

    attn = F.softmax(scores, dim=-1)
    context_vec = attn @ V

    ff = torch.relu(context_vec @ w1 + b1)
    context_vec = ff @ w2 + b2

    logits = context_vec[-1] @ wo
    probs = F.softmax(logits, dim=0)
    #predict next token and add it to the window to generate next token
    next_id = torch.multinomial(probs, 1) #take the token with highest probability
    context = torch.cat([context[1:], next_id])

generated = " ".join(itos[i.item()] for i in context)
print(generated)
