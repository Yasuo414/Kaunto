import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import datasets
import configparser
import torchvision.transforms

config = configparser.ConfigParser()
config.read("settings.ini")
datas = config.get("Settings", "Dataset")
name_of_model = config.get("Settings", "Model_name")
version_of_model = config.get("Settings", "Version")
original_authors = "Yasuo4144 and ChatGPT"
print(f"Original authors: {original_authors}")
chatgpt = "https://chat.openai.com"
yasuo4144 = "https://github.com/Yasuo414"
author = config.get("Settings", "Author")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 192)
        self.fc4 = nn.Linear(192, 3 * 1920 * 1080)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(3 * 1920 * 1080, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

generator = Generator()
discriminator = Discriminator()

generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

num_epochs = 1000

dataset = # Your dataset class in this syntax: YourDatasetClass(root="<root of your dataset>", transform=transform)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])
batch_size = # Your batch size number
snuffle = #True/False choose please
num_workers = #Your count of working string

data_loader = DataLoader(dataset, batch_size=batch_size, snuffle=snuffle, num_workers=num_workers, pin_memory=True, drop_last=True)

for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(data_loader):  # Nahraďte 'data_loader' daty ze svého trénovacího datasetu
        # Nastavení gradientů na nulu
        generator.zero_grad()
        discriminator.zero_grad()

        # Trénink diskriminátoru na reálných datech
        real_images = real_images.view(real_images.size(0), -1)  # Zploštění obrázku
        real_labels = torch.ones(real_images.size(0), 1)
        real_outputs = discriminator(real_images)
        real_loss = criterion(real_outputs, real_labels)
        real_loss.backward()

        # Generování falešných obrázků a trénink diskriminátoru na falešných datech
        noise = torch.randn(real_images.size(0), 100)
        fake_images = generator(noise)
        fake_labels = torch.zeros(real_images.size(0), 1)
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_loss.backward()

        # Celkový loss diskriminátoru
        discriminator_loss = real_loss + fake_loss

        # Aktualizace vah diskriminátoru
        discriminator_optimizer.step()

        # Trénink generátoru
        generator.zero_grad()
        discriminator.zero_grad()

        noise = torch.randn(real_images.size(0), 100)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images)
        generator_loss = criterion(fake_outputs, real_labels)  # Chceme, aby generátor přesvědčil diskriminátor, že jeho obrázky jsou reálné

        # Aktualizace vah generátoru
        generator_loss.backward()
        generator_optimizer.step()

        # Výpis průběhu tréninku
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(data_loader)}] Discriminator Loss: {discriminator_loss.item():.4f}, Generator Loss: {generator_loss.item():.4f}")

    # Uložení vygenerovaného obrázku na konci epochy (pro kontrolu)
    if (epoch + 1) % 10 == 0:
        generated_image = fake_images[0].view(3, 1920, 1080)  # Přizpůsobení tvaru generovaného obrázku
        save_image(generated_image, f"generated_image_epoch{epoch + 1}.png")
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")