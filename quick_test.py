import torch
from train_echo import Echo, echo_Config
import tiktoken

# small config for quick test
cfg = echo_Config(block_size=32, vocab_size=50257, n_layer=2, n_head=4, n_embd=128)
model = Echo(cfg)
model.eval()

B, T = 2, 8
x = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)
with torch.no_grad():
    logits, loss = model(x)
print('logits shape:', logits.shape)
print('loss:', loss)

# simple generation: greedy sampling for a few steps
enc = tiktoken.get_encoding('gpt2')
start = enc.encode("Hello")
start_t = torch.tensor([start], dtype=torch.long)
# pad/truncate to fit model block size
if start_t.size(1) > cfg.block_size:
    start_t = start_t[:, -cfg.block_size:]

with torch.no_grad():
    out = start_t
    for _ in range(10):
        logits, _ = model(out)
        next_logits = logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        out = torch.cat([out, next_token], dim=1)

tokens = out[0].tolist()
print('generated token ids:', tokens)
print('decoded:', enc.decode(tokens))
