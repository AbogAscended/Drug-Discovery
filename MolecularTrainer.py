from CharRNN import CharRNNV2
from Generation import *
from DataLoader import *
import torch
from onehotencoder import OneHotEncoder
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
torch.set_float32_matmul_precision("high")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

endecode = OneHotEncoder()
vocab_size = OneHotEncoder.get_vocab_size(self = endecode)
num_layers = 3
n_gram = 2
dropped_out = 0.2
num_workers = 5
hidden_size = 1024
learning_rate = 1e-4
num_epochs = 20
batch_size = 128
temp = 1
p = 1

def main():
    file_paths = [f'data/seqs_len{i}.txt' for i in range(18, 52)]
    data = Data(file_paths, endecode, n_gram, batch_size, num_workers, num_epochs)
    train_loader, val_loader, total_steps, warmup_steps = data.get_loaders()

    charRNN = CharRNNV2(
       vocab_size, num_layers, n_gram,
       total_steps=total_steps, warmup_steps=warmup_steps,
       lr=learning_rate, hidden_size=hidden_size, dropout=dropped_out
    )

    trainer = Trainer(
        default_root_dir="lr_find_ckpts",
        max_epochs=num_epochs,
        accelerator="cuda",
        precision="16-mixed",
        gradient_clip_val=5.0,
        logger=TensorBoardLogger("tb_logs", name="char_rnn"),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", patience=5),
            RichProgressBar()
        ],
        profiler="pytorch",
    )

    tuner = Tuner(trainer)
    tuner.lr_find(
        charRNN,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        min_lr=1e-6,
        max_lr=1.0,
        early_stop_threshold=None,
        update_attr=True,
    )

    trainer.fit(charRNN, train_loader, val_loader)
    torch.save(charRNN.state_dict(), "Models/charRNNv2-gram.pt")
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()