from lightning.pytorch.callbacks import LearningRateMonitor
from CharRNN import CharRNN
from DataLoader import *
import torch
from onehotencoder import OneHotEncoder
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import torch._dynamo
torch._dynamo.config.cache_size_limit = 32
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


endecode = OneHotEncoder()
vocab_size = OneHotEncoder.get_vocab_size(self = endecode)
num_layers = 3
n_gram = 1
dropped_out = 0.5
learning_rate = 1e-6
num_epochs = 5
kl_epochs = 2
batch_size = 128
hidden_size = 1024
num_workers = 5

def main():
    file_paths = [f'data/seqs_len{i}.txt' for i in range(18, 52)]
    file_paths_test = [f'data/seqs_len{i}_test.txt' for i in range(22, 49)]
    data = Data(file_paths, file_paths_test, endecode, n_gram, batch_size, num_workers, num_epochs)
    train_loader, val_loader, total_steps, warmup_steps = data.get_loaders()
    charRNN = CharRNN(
        vocab_size,
        num_layers,
        n_gram,
        dropped_out,
        learning_rate,
        kl_epochs,
        hidden_size,
        warmup_steps,
        total_steps,
    )
    charRNN = torch.compile(charRNN)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        filename='check/char-rnn-{epoch:02d}-{val_loss:.2f}'
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
            EarlyStopping(monitor="val_loss", patience=1),
            RichProgressBar()
        ],
        profiler=None,
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
    torch.save(charRNN.state_dict(), "Models/charRNNNoFlow1-gram.pt")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
