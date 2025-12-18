#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

DATA_ROOT = Path("full_512x128")
OUT_SAMPLES = Path("samples_dcgan_512x128")
OUT_CHECKPOINTS = Path("checkpoints_dcgan_512x128")

OUT_SAMPLES.mkdir(exist_ok=True, parents=True)
OUT_CHECKPOINTS.mkdir(exist_ok=True, parents=True)

image_size = (128, 512)   # (H, W)
nz = 100                  # latent dim
nc = 1                    # grayscale

ngf = 32                  # generator base channels (smaller than before)
ndf = 32                  # discriminator base channels

batch_size = 8            # safer on CPU
num_epochs = 200
lr = 1e-4                 # lower LR for stability
beta1 = 0.5

device = torch.device("cpu")
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
            transforms.Normalize((0.5,), (0.5,)),    # [-1,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("L")
        img = img.resize((image_size[1], image_size[0]), Image.BICUBIC)
        return self.transform(img)

dataset = SonarDataset(DATA_ROOT)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=2, pin_memory=False)

print(f"Dataset size: {len(dataset)} images")

# ---------------------------------------------------------
# Generator  ->  1 x 128 x 512
# ---------------------------------------------------------

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Z: nz x 1 x 1
            nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias=False),   # 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, (4, 8), (2, 4), (1, 2), bias=False),   # 32 x 32 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, (4, 8), (2, 4), (1, 2), bias=False),   # 16 x 64 x 256
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, nc, 4, 2, 1, bias=False),    # 1 x 128 x 512
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

# ---------------------------------------------------------
# Discriminator  <-  1 x 128 x 512
# ---------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, 16, 4, 2, 1, bias=False),        # 16 x 64 x 256
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, (4, 8), (2, 4), (1, 2), bias=False),  # 32 x 32 x 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, (4, 8), (2, 4), (1, 2), bias=False),  # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),       # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),      # 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),        # 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(-1, 1)

netG = Generator().to(device)
netD = Discriminator().to(device)

# -------------------------------------------------------
# RESUME TRAINING FROM A GIVEN EPOCH
# Set resume_epoch to the checkpoint you want to load.
# Example: to load netG_epoch_020.pth, set resume_epoch = 20
# -------------------------------------------------------
resume_epoch = 100  # change this to the desired epoch number

if resume_epoch > 0:
    ckptG = f"checkpoints_dcgan_512x128/netG_epoch_{resume_epoch:03d}.pth"
    ckptD = f"checkpoints_dcgan_512x128/netD_epoch_{resume_epoch:03d}.pth"
    print(f"Resuming training from epoch {resume_epoch}")
    netG.load_state_dict(torch.load(ckptG, map_location=device))
    netD.load_state_dict(torch.load(ckptD, map_location=device))
else:
    # if starting fresh → initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    netG.apply(weights_init)
    netD.apply(weights_init)

# weight init
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
# Loss & optimizers
# ---------------------------------------------------------

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(8, nz, 1, 1, device=device)

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

start_epoch = resume_epoch
for epoch in range(start_epoch, num_epochs):
    for i, real_batch in enumerate(dataloader):
        b_size = real_batch.size(0)
        real_batch = real_batch.to(device)

        # label smoothing for real
        real_labels = torch.full((b_size, 1), 0.9, device=device)
        fake_labels = torch.full((b_size, 1), 0.0, device=device)

        # small noise on inputs to D (regularisation)
        real_batch = real_batch + 0.05 * torch.randn_like(real_batch)

        # 1) Update D
        netD.zero_grad()
        output_real = netD(real_batch)
        lossD_real = criterion(output_real, real_labels)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        fake_images_noisy = fake_images + 0.05 * torch.randn_like(fake_images)
        output_fake = netD(fake_images_noisy.detach())
        lossD_fake = criterion(output_fake, fake_labels)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # 2) Update G
        netG.zero_grad()
        output_fake_for_G = netD(fake_images_noisy)
        lossG = criterion(output_fake_for_G, real_labels)  # want D(G(z)) -> real
        lossG.backward()
        optimizerG.step()

        if i % 20 == 0:
            print(f"[{epoch+1}/{num_epochs}] step {i}/{len(dataloader)} "
                  f"Loss_D: {lossD.item():.4f}  Loss_G: {lossG.item():.4f}")

    # Save samples each epoch
    with torch.no_grad():
        fake = netG(fixed_noise).cpu()
        fake = (fake + 1) / 2.0   # [-1,1] -> [0,1]
        for k in range(fake.size(0)):
            img = fake[k, 0]
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(OUT_SAMPLES / f"epoch_{epoch+1:03d}_sample_{k}.png")

    torch.save(netG.state_dict(), OUT_CHECKPOINTS / f"netG_epoch_{epoch+1:03d}.pth")
    torch.save(netD.state_dict(), OUT_CHECKPOINTS / f"netD_epoch_{epoch+1:03d}.pth")
