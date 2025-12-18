#!/usr/bin/env python3
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

DATA_ROOT = Path("full_1024x256")  # folder with resized sonar images
OUT_SAMPLES = Path("samples_dcgan")
OUT_CHECKPOINTS = Path("checkpoints_dcgan")

OUT_SAMPLES.mkdir(exist_ok=True, parents=True)
OUT_CHECKPOINTS.mkdir(exist_ok=True, parents=True)

image_size = (256, 1024)  # (H, W)
nz = 100                  # latent vector size
ngf = 32                  # generator feature size
ndf = 32                  # discriminator feature size
nc = 1                    # channels (grayscale)

batch_size = 16
num_epochs = 100
lr = 5e-5
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class SonarDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.png"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),                   # [0,1]
            transforms.Normalize((0.5,), (0.5,)),    # -> [-1,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("L")
        img = img.resize((image_size[1], image_size[0]), Image.BICUBIC)
        img = self.transform(img)
        return img

dataset = SonarDataset(DATA_ROOT)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=2, pin_memory=True)

print(f"Dataset size: {len(dataset)} images")

# ---------------------------------------------------------
# DCGAN Generator (outputs 1×256×1024)
# ---------------------------------------------------------

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input Z: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),   # -> 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # -> 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # -> 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # -> 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # stretch more in width than height:
            nn.ConvTranspose2d(64, 32, (4, 8), (2, 4), (1, 2), bias=False),  # -> 32 x 64 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, (4, 8), (2, 4), (1, 2), bias=False),  # -> 16 x 128 x 512
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, nc, 4, 2, 1, bias=False),    # -> 1 x 256 x 1024
            nn.Tanh(),                                          # [-1,1]
        )

    def forward(self, x):
        return self.main(x)


# ---------------------------------------------------------
# DCGAN Discriminator (inputs 1×256×1024)
# ---------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input: 1 x 256 x 1024
            nn.Conv2d(nc, 16, 4, 2, 1, bias=False),             # -> 16 x 128 x 512
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, (4, 8), (2, 4), (1, 2), bias=False),  # -> 32 x 64 x 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, (4, 8), (2, 4), (1, 2), bias=False),  # -> 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),            # -> 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),           # -> 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),           # -> 512 x 4 x 4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),             # -> 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(-1, 1)


netG = Generator().to(device)
netD = Discriminator().to(device)

# weight init (standard DCGAN)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# ---------------------------------------------------------
# Loss and optimizers
# ---------------------------------------------------------

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(8, nz, 1, 1, device=device)  # for monitoring progress

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

for epoch in range(num_epochs):
    for i, real_batch in enumerate(dataloader):
        b_size = real_batch.size(0)
        real_batch = real_batch.to(device)

        # Real and fake labels
        real_labels = torch.full((b_size, 1), 0.9, device=device)
        fake_labels = torch.full((b_size, 1), 0.0, device=device)

        # ---------------------------
        # 1. Update D: maximize log(D(x)) + log(1 - D(G(z)))
        # ---------------------------
        netD.zero_grad()

        # Real
        output_real = netD(real_batch)
        lossD_real = criterion(output_real, real_labels)

        # Fake
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, fake_labels)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # ---------------------------
        # 2. Update G: maximize log(D(G(z)))
        # ---------------------------
        netG.zero_grad()
        output_fake_for_G = netD(fake_images)
        lossG = criterion(output_fake_for_G, real_labels)  # want D(G(z)) -> 1
        lossG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"[{epoch+1}/{num_epochs}] "
                  f"step {i}/{len(dataloader)}  "
                  f"Loss_D: {lossD.item():.4f}  Loss_G: {lossG.item():.4f}")

    # Save samples after each epoch
    with torch.no_grad():
        fake = netG(fixed_noise).cpu()
        # de-normalise from [-1,1] to [0,1]
        fake = (fake + 1) / 2.0
        for k in range(fake.size(0)):
            img = fake[k, 0]  # single-channel
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(OUT_SAMPLES / f"epoch_{epoch+1:03d}_sample_{k}.png")

    # Save checkpoints
    torch.save(netG.state_dict(), OUT_CHECKPOINTS / f"netG_epoch_{epoch+1:03d}.pth")
    torch.save(netD.state_dict(), OUT_CHECKPOINTS / f"netD_epoch_{epoch+1:03d}.pth")
