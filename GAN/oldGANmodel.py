# Install required libraries with specific versions
!pip install rdkit-pypi==2022.9.5
!pip install torch==1.13.1
!pip install numpy==1.23.5
!pip install pandas scikit-learn tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import os
from google.colab import drive
from torch.cuda.amp import GradScaler, autocast
import logging

# Suppress RDKit warnings to reduce spamming
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
            raise ValueError(f"No 'SMILES' column found in {file_path}. Columns found: {df.columns}")
        smiles_list = df['SMILES'].dropna().astype(str).tolist()
        print(f"First 5 SMILES from {file_path}:")
        print(smiles_list[:5])
        valid_smiles = []
        invalid_smiles = []
        for i, s in enumerate(smiles_list, 1):
            s = s.strip()
            if not s:
                invalid_smiles.append((i, s, "Empty string"))
                continue
            try:
                mol = Chem.MolFromSmiles(s, sanitize=True)
                if mol:
                    valid_smiles.append(s)
                else:
                    invalid_smiles.append((i, s, "Invalid SMILES"))
            except Exception as e:
                invalid_smiles.append((i, s, f"Parse error: {str(e)}"))
        if invalid_smiles:
            print(f"Warning: {len(invalid_smiles)} invalid SMILES in {file_path}. Examples:")
            for line_num, invalid, reason in invalid_smiles[:5]:
                print(f"Line {line_num}: {invalid} ({reason})")
        if not valid_smiles:
            raise ValueError(f"No valid SMILES found in {file_path}")
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
class SMILESAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_len):
        super(SMILESAutoencoder, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder_rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.encoder_fc = nn.Linear(hidden_dim * 2, latent_dim)
        self.decoder_rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_dim, vocab_size)
    def encode(self, x):
        embedded = self.embedding(x)
        _, hidden = self.encoder_rnn(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        latent = self.encoder_fc(hidden)
        return latent
    def decode(self, z):
        batch_size = z.size(0)
        hidden = torch.zeros(1, batch_size, self.decoder_rnn.hidden_size).to(z.device)
        input_seq = torch.zeros(batch_size, 1, self.decoder_rnn.input_size).to(z.device)
        outputs = []
        for _ in range(self.max_len):
            output, hidden = self.decoder_rnn(input_seq, hidden)
            logits = self.decoder_fc(output.squeeze(1))
            outputs.append(logits)
            input_seq = self.embedding(torch.argmax(logits, dim=1).unsqueeze(1))
        return torch.stack(outputs, dim=1)
    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output logits
        )
    def forward(self, z):
        return self.net(z)

# --- Training Functions ---
def validate_autoencoder(model, dataloader, idx_to_char, device, temperature=0.7):
    """Validate autoencoder by checking reconstructed SMILES validity with sampling"""
    model.eval()
    valid_count = 0
    total_count = 0
    example_smiles = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon, _ = model(batch)
            # Apply temperature sampling
            probs = torch.softmax(recon / temperature, dim=-1)
            preds = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.size(0), probs.size(1))
            preds = preds.cpu().numpy()
            for pred in preds:
                s = ''.join(idx_to_char[idx] for idx in pred if idx != 0)
                if len(example_smiles) < 5:  # Collect first 5 SMILES for debugging
                    example_smiles.append(s)
                try:
                    mol = Chem.MolFromSmiles(s, sanitize=True)
                    if mol:
                        valid_count += 1
                except:
                    pass
                total_count += 1
    validity = valid_count / total_count if total_count > 0 else 0
    print(f"Autoencoder Validation: {valid_count}/{total_count} valid SMILES ({validity:.4f})")
    print("Example reconstructed SMILES:")
    for i, s in enumerate(example_smiles, 1):
        print(f"{i}. {s}")
    return validity

def train_autoencoder(model, dataloader, epochs, device, idx_to_char):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with autocast():
                recon, _ = model(batch)
                loss = criterion(recon.view(-1, recon.size(-1)), batch.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Autoencoder Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    # Validate after training
    validate_autoencoder(model, dataloader, idx_to_char, device)

def train_gan(generator, discriminator, autoencoder, dataloader, epochs, device, latent_dim):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    autoencoder.eval()
    for epoch in range(epochs):
        d_loss_total, g_loss_total = 0, 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            batch_size = batch.size(0)
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)
            # Train Discriminator
            d_optimizer.zero_grad()
            with torch.no_grad():
                real_latent = autoencoder.encode(batch)
            with autocast():
                real_output = discriminator(real_latent)
                d_loss_real = criterion(real_output, real_label)
                noise = torch.randn(batch_size, latent_dim).to(device)
                fake_latent = generator(noise)
                fake_output = discriminator(fake_latent)
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
                fake_latent = generator(noise)
                fake_output = discriminator(fake_latent)
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
def generate_smiles(generator, autoencoder, num_samples, latent_dim, idx_to_char, device, temperature=0.7):
    """Generate SMILES strings using the trained GAN model with temperature sampling."""
    generator.eval()
    autoencoder.eval()
    smiles = []
    invalid_smiles = []
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        latent_vectors = generator(noise)
        outputs = autoencoder.decode(latent_vectors)
        # Apply temperature sampling
        probs = torch.softmax(outputs / temperature, dim=-1)
        preds = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.size(0), probs.size(1))
        preds = preds.cpu().numpy()
        for pred in preds:
            s = ''.join(idx_to_char[idx] for idx in pred if idx != 0)
            try:
                mol = Chem.MolFromSmiles(s, sanitize=True)
                if mol:
                    smiles.append(Chem.MolToSmiles(mol))
                else:
                    invalid_smiles.append(s)
            except:
                invalid_smiles.append(s)
        if invalid_smiles:
            print(f"Warning: {len(invalid_smiles)} generated SMILES are invalid. Examples:")
            print(invalid_smiles[:5])
        print(f"Generated {len(smiles)} valid SMILES")
        return smiles

def evaluate_generated_smiles(generated_smiles, test_smiles):
    """Evaluate generated SMILES using RDKit metrics."""
    valid_smiles = [s for s in generated_smiles if Chem.MolFromSmiles(s, sanitize=True)]
    validity = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0
    unique_smiles = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0
    metrics = {"validity": validity, "uniqueness": unique_smiles}
    print(f"Validity: {validity:.4f}")
    print(f"Uniqueness: {unique_smiles:.4f}")
    return metrics

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
    ae_epochs = 20
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
    autoencoder = SMILESAutoencoder(vocab_size, embedding_dim, hidden_dim, latent_dim, max_len).to(device)
    generator = Generator(latent_dim, hidden_dim).to(device)
    discriminator = Discriminator(latent_dim, hidden_dim).to(device)
    # Train autoencoder
    print("Training Autoencoder...")
    train_autoencoder(autoencoder, dataloader, ae_epochs, device, idx_to_char)
    # Train GAN
    print("Training GAN...")
    train_gan(generator, discriminator, autoencoder, dataloader, gan_epochs, device, latent_dim)
    # Generate SMILES
    print("Generating SMILES...")
    generated_smiles = generate_smiles(generator, autoencoder, 10000, latent_dim, idx_to_char, device, temperature=0.7)
    # Evaluate generated SMILES
    print("Evaluating Generated SMILES...")
    evaluate_generated_smiles(generated_smiles, test_smiles)
    # Save generated SMILES
    output_path = "/content/drive/MyDrive/Data/generated_smiles.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(generated_smiles))
    print(f"Generated SMILES saved to {output_path}")

if __name__ == "__main__":
    main()
