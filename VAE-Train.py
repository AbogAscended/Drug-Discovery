import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Hyperparameters
latent_dim = 8  # Dimension of latent space
embedding_dim = 768  # Word embedding dimension
max_len = 200  # Max length of input sequence
vocab_size = 30522  # Vocabulary size for bert-base-uncased
batch_size = 64
epochs = 10

# Load IMDB dataset using Hugging Face datasets
dataset = load_dataset('stanfordnlp/imdb')
train_data = dataset['train']
test_data = dataset['test']

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and pad sequences
def tokenize_and_pad(data, max_len=max_len):
    tokenized = tokenizer(data['text'], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    return tokenized['input_ids']

# Prepare data
x_train = tokenize_and_pad(train_data)
x_test = tokenize_and_pad(test_data)

# Create DataLoader
train_dataset = TensorDataset(x_train, x_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(x_test, x_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define VAE Model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.fc_z = nn.Linear(latent_dim, 64)
        self.lstm3 = nn.LSTM(64, 64, batch_first=True)
        self.lstm4 = nn.LSTM(64, 128, batch_first=True)
        self.fc_out = nn.Linear(128, vocab_size)
    
    def encode(self, x):
        h = self.embedding(x)
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        mu = self.fc_mu(h[:, -1, :])
        logvar = self.fc_logvar(h[:, -1, :])
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_z(z)
        h = h.unsqueeze(1).repeat(1, max_len, 1)
        h, _ = self.lstm3(h)
        h, _ = self.lstm4(h)
        return self.fc_out(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss Function
def vae_loss(x_decoded, x, mu, logvar):
    reconstruction_loss = nn.CrossEntropyLoss()(x_decoded.view(-1, vocab_size), x.view(-1))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_loss

# Initialize Model, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.AdamW(model.parameters())

# Training Loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}")

# Test Loop
# def test():
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for data, _ in test_loader:
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += vae_loss(recon_batch, data, mu, logvar).item()
#     print(f"Test Loss: {test_loss / len(test_loader.dataset)}")

# Train Model
for epoch in range(1, epochs + 1):
    train(epoch)
    #test()

# Generate Text
def generate_text():
    model.eval()
    z_sample = torch.randn(10, latent_dim).to(device)
    generated_seq = model.decode(z_sample)
    generated_text = " ".join([str(torch.argmax(word).item()) for word in generated_seq[0]])
    return generated_text

print("Generated text:", generate_text())