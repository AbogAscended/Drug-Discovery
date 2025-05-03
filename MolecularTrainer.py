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
n_gram = 1
dropped_out = 0.2
num_workers = 5
hidden_size = 1024
learning_rate = 5e-4
num_epochs = 200
batch_size = 128
temp = 1
p = 1


file_paths = [f'data/seqs_len{i}.txt' for i in range(18, 52)]
Data = Data(filepaths=file_paths,encoder=endecode,n_gram=n_gram,batch_size=batch_size,num_workers=num_workers,num_epochs=num_epochs)
train_loader, val_loader, total_steps, warmup_steps = Data.get_loaders()


charRNN = CharRNNV2(vocab_size, num_layers, n_gram, total_steps, warmup_steps, learning_rate, hidden_size, dropped_out).to(device)
trainer = Trainer(
    max_epochs=200,
    accelerator="cuda",
    precision='16-mixed',
    gradient_clip_val=5.0,
    logger=TensorBoardLogger("tb_logs", name="char_rnn"),
    callbacks=[
        ModelCheckpoint(monitor="val_loss", mode="min"),
        EarlyStopping(monitor="val_loss", patience=5),
        RichProgressBar(),
    ],
    profiler="pytorch",
    enable_progress_bar=True,
)

tuner = Tuner(trainer)
lr_find_results = tuner.lr_find(
    charRNN,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    min_lr=1e-6,
    max_lr=1.0,
    early_stop_threshold=None,
    update_attr=True,
)
trainer.fit(charRNN, train_loader, val_loader)


torch.save(charRNN.state_dict(), "Models/charRNNv1-gram.pt")
filepath = 'data/GRUOnly1P1-gram.txt'
generator = Generator(charRNN, endecode, vocab_size, n_gram, p, temp)
generator.generate(filepath)