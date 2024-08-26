import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb

class DeepCraft(nn.Module):
    def __init__(self, num_block_types, latent_dim=128, embedding_dim=32):
        super().__init__()

        self.embedding = nn.Embedding(num_block_types, embedding_dim)

        self.encoder = nn.Sequential(
            nn.Conv3d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.flatten_size = 256 * 6 * 6 * 6

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)

        self.fc_decoder = nn.Linear(latent_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, num_block_types, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose3d(num_block_types, num_block_types, kernel_size=3, stride=1, padding=1)
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)

    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, 256, 6, 6, 6)
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class MinecraftDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data['arr_0']).int()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.cross_entropy(recon_x, x.long(), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = vae_loss(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        running_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({"Batch loss": f"{loss.item():.4f}", "Avg loss": f"{running_loss:.4f}"})
        wandb.log({
            "Epoch": epoch + 1,
            "Batch": batch_idx,
            "Batch Loss": loss.item(),
            "Running Loss": running_loss
        })

        # TODO: clean that shit
        if (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

    return total_loss / len(dataloader)

# TODO: add CLI arguments possibility and way to resume training
def main():
    if os.path.exists('mat_toid_mapping.json'):
        # read num_block_types from there
        with open('mat_toid_mapping.json', 'r') as f:
            mat_toid_mapping = json.load(f)
            num_block_types = len(mat_toid_mapping)
    else:
        num_block_types = 437

    wandb.init(project="deepcraft", config={
        "num_block_types": num_block_types,
        "batch_size": 16,
        "num_epochs": 50,
        "learning_rate": 0.0003,
        "latent_dim": 128,
        "embedding_dim": 32
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_block_types = wandb.config.num_block_types
    batch_size = wandb.config.batch_size
    num_epochs = wandb.config.num_epochs
    learning_rate = wandb.config.learning_rate
    latent_dim = wandb.config.latent_dim
    embedding_dim = wandb.config.embedding_dim

    data = np.load('builds.npz')
    dataset = MinecraftDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = DeepCraft(num_block_types, latent_dim=latent_dim, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = train(model, dataloader, optimizer, device, epoch)
        wandb.log({
            "Epoch": epoch + 1,
            "Epoch Loss": epoch_loss,
        })

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'deepcraft_epoch_{epoch+1}.pth')
            wandb.save(f'deepcraft_epoch_{epoch+1}.pth')

    torch.save(model.state_dict(), 'deepcraft_final.pth')
    wandb.save('deepcraft_final.pth')

    wandb.finish()

if __name__ == "__main__":
    main()