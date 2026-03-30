import logging
import os
from tokenization import SimpleBPE
import torch
from data_prep.prepare import prepare_data
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
vocab_size = 1000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


with open("input.txt","r") as f:
    text = f.read()


def log_setup():
    log_dir= "logs"
    os.makedirs(log_dir,exist_ok=True)

    file_name = "gpt.log"

    full_path = os.path.join(log_dir,file_name)

    logging.basicConfig(
        level=logging.INFO,
        format= "%(asctime)s - %(levelname)s %(message)s",
        handlers= [logging.FileHandler(full_path),logging.StreamHandler()]
    )

    logging.info("Logging initialize successfully")


def main():

    log_setup()
    prepare_data("input.txt","data",train_file_name="train.bin",val_file_name='val.bin',vocab_size=vocab_size)



if __name__ == "__main__":
    main()