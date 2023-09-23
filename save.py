import torch
import configparser

config = configparser.ConfigParser()
config.read("settings.ini")
name = config.get("Settings", "Model_name")
ver = config.get("Settings", "Version")
author = config.get("Settings", "Author")

generator_weights = torch.load("generator.pth")
discriminator_weights = torch.load("discriminator.pth")

combined_weights = {
    "generator": generator_weights,
    "discriminator": discriminator_weights
}

torch.save(combined_weights, f"Kaunto-{name}-{ver}-{author}.pth")