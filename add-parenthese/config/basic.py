# train a miniature character-level model
# consisting of fixed strings consisting of a few characters in alphabetical order.

out_dir = 'out/basic'
eval_interval = 50  # keep frequent because we'll overfit
eval_iters = 20
log_interval = 1

# always save checkpoint so sampling works
always_save_checkpoint = True

wandb_log = True
wandb_project = 'add-parentheses-basic'
wandb_run_name = 'first run'

dataset = 'basic'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64

# slightly larger GPT model
n_layer = 4
n_head = 4
n_embd = 128  # n_embd % n_head == 0
dropout = 0.0

learning_rate = 1e-3
max_iters = 500
lr_decay_iters = 500
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

device = 'cpu'
compile = False

########################################################################
### sampling-specific params
# temperature and top_k for slightly better outputs
temperature = 0.7
top_k = 20
num_samples = 1
max_new_tokens = 100
seed = 2345
