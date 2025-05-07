# Install required libraries with specific versions
!pip install rdkit-pypi==2022.9.5
!pip install torch==1.13.1
!pip install numpy==1.23.5
!pip install pandas scikit-learn tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import os
from google.colab import drive
from torch.cuda.amp import GradScaler, autocast
import logging

# Suppress RDKit warnings
logging.getLogger('rdkit').setLevel(logging.ERROR)

# Mount Google Drive
drive.mount('/content/drive')

# --- Verify File Existence ---
def check_files(train_path, test_path):
    """Check if train and test files exist"""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")
    print("Both train and test files found!")

# --- Data Loading and Preprocessing ---
def load_smiles(file_path):
    """Load SMILES strings from a CSV-like .txt file with a 'SMILES' column"""
    try:
        df = pd.read_csv(file_path, sep=',', on_bad_lines='skip')
        if 'SMILES' not in df.columns:
            raise ValueError(f"No 'SMILES' column found in {file_path}")
        smiles_list = df['SMILES'].dropna().astype(str).tolist()
        valid_smiles = [s.strip() for s in smiles_list if Chem.MolFromSmiles(s.strip(), sanitize=True)]
        print(f"Loaded {len(valid_smiles)} valid SMILES from {file_path}")
        return valid_smiles
    except Exception as e:
        raise ValueError(f"Error loading SMILES from {file_path}: {str(e)}")

def build_vocab(smiles_list):
    """Build vocabulary from SMILES strings"""
    chars = sorted(set(''.join(smiles_list)))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 0 for padding
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(chars) + 1  # +1 for padding

def tokenize_smiles(smiles_list, char_to_idx, max_len):
    """Convert SMILES strings to tokenized sequences"""
    tokenized = []
    for s in smiles_list:
        seq = [char_to_idx[c] for c in s]
        seq = seq[:max_len] + [0] * (max_len - len(seq))  # Pad or truncate
        tokenized.append(seq)
    return np.array(tokenized)

class SMILESDataset(Dataset):
    def __init__(self, tokenized_smiles):
        self.data = torch.tensor(tokenized_smiles, dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- Model Definitions ---
class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_len):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.initial_hidden = nn.Linear(latent_dim, hidden_dim)

    def forward(self, z, tau=1.0):
        batch_size = z.size(0)
        h0 = self.initial_hidden(z).unsqueeze(0)  # (1, batch_size, hidden_dim)
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        input_emb = self.embedding(start_token)  # (batch_size, 1, embedding_dim)
        outputs = []
        hidden = h0
        for t in range(self.max_len):
            output, hidden = self.rnn(input_emb, hidden)
            logits = self.output_layer(output.squeeze(1))  # (batch_size, vocab_size)
            soft_one_hot = F.gumbel_softmax(logits, tau=tau, hard=True)
            input_emb = torch.matmul(soft_one_hot, self.embedding.weight).unsqueeze(1)
            outputs.append(soft_one_hot)
        outputs = torch.stack(outputs, dim=1)  # (batch_size, max_len, vocab_size)
        return outputs

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, sequences):
        embedded = self.embedding(sequences)  # (batch_size, max_len, embedding_dim)
        _, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        logits = self.output_layer(hidden)  # (batch_size, 1)
        return logits

# --- Training Functions ---
def train_gan(generator, discriminator, dataloader, epochs, device, latent_dim, vocab_size):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    for epoch in range(epochs):
        d_loss_total, g_loss_total = 0, 0
        for batch in dataloader:
            batch = batch.to(device)
            batch_size = batch.size(0)
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            with autocast():
                real_output = discriminator(batch)
                d_loss_real = criterion(real_output, real_label)
                noise = torch.randn(batch_size, latent_dim).to(device)
                fake_sequences_one_hot = generator(noise)
                fake_sequences = torch.argmax(fake_sequences_one_hot, dim=-1)
                fake_output = discriminator(fake_sequences)
                d_loss_fake = criterion(fake_output, fake_label)
                d_loss = d_loss_real + d_loss_fake
            scaler.scale(d_loss).backward()
            scaler.unscale_(d_optimizer)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler.step(d_optimizer)
            scaler.update()

            # Train Generator
            g_optimizer.zero_grad()
            with autocast():
                noise = torch.randn(batch_size, latent_dim).to(device)
                fake_sequences_one_hot = generator(noise)
                fake_sequences = torch.argmax(fake_sequences_one_hot, dim=-1)
                fake_output = discriminator(fake_sequences)
                g_loss = criterion(fake_output, real_label)
            scaler.scale(g_loss).backward()
            scaler.unscale_(g_optimizer)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler.step(g_optimizer)
            scaler.update()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
        print(f"GAN Epoch {epoch + 1}/{epochs}, D Loss: {d_loss_total / len(dataloader):.4f}, G Loss: {g_loss_total / len(dataloader):.4f}")

# --- Generation and Evaluation ---
def generate_smiles(generator, num_samples, latent_dim, idx_to_char, device, tau=1.0):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        fake_sequences_one_hot = generator(noise, tau=tau)
        fake_sequences = torch.argmax(fake_sequences_one_hot, dim=-1)
        smiles = []
        for seq in fake_sequences:
            s = ''.join(idx_to_char[idx.item()] for idx in seq if idx != 0)
            smiles.append(s)
        return smiles

def evaluate_generated_smiles(generated_smiles):
    valid_smiles = [s for s in generated_smiles if Chem.MolFromSmiles(s, sanitize=True)]
    validity = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0
    unique_smiles = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0
    print(f"Validity: {validity:.4f}")
    print(f"Uniqueness: {unique_smiles:.4f}")
    return {"validity": validity, "uniqueness": unique_smiles}

# --- Main Execution ---
def main():
    # Define file paths
    train_path = "/content/drive/MyDrive/Data/train.txt"
    test_path = "/content/drive/MyDrive/Data/test.txt"
    # Verify files
    check_files(train_path, test_path)
    # Hyperparameters
    max_len = 100
    embedding_dim = 64
    hidden_dim = 256
    latent_dim = 128
    batch_size = 64
    gan_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load and preprocess data
    train_smiles = load_smiles(train_path)
    test_smiles = load_smiles(test_path)
    all_smiles = train_smiles
    char_to_idx, idx_to_char, vocab_size = build_vocab(all_smiles)
    tokenized_smiles = tokenize_smiles(all_smiles, char_to_idx, max_len)
    dataset = SMILESDataset(tokenized_smiles)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Initialize models
    generator = Generator(vocab_size, embedding_dim, hidden_dim, latent_dim, max_len).to(device)
    discriminator = Discriminator(vocab_size, embedding_dim, hidden_dim).to(device)
    # Train GAN
    print("Training GAN...")
    train_gan(generator, discriminator, dataloader, gan_epochs, device, latent_dim, vocab_size)
    # Generate SMILES
    print("Generating SMILES...")
    generated_smiles = generate_smiles(generator, 10000, latent_dim, idx_to_char, device)
    # Evaluate generated SMILES
    print("Evaluating Generated SMILES...")
    evaluate_generated_smiles(generated_smiles)
    # Save generated SMILES
    output_path = "/content/drive/MyDrive/Data/generated_smiles.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(generated_smiles))
    print(f"Generated SMILES saved to {output_path}")

if __name__ == "__main__":
    main()