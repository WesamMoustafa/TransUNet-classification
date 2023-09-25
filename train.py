import argparse
import os

import lightning as l
import torch
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as tf

from model_no_decoder import CONFIGS, VisionTransformer

wandb.login()


def model_size(model):
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


train_dataset = datasets.CIFAR10(
    root="./data", train=True, transform=tf.ToTensor(), download=True
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, transform=tf.ToTensor(), download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=False,
    num_workers=os.cpu_count(),  # type: ignore
)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    num_workers=os.cpu_count(),  # type: ignore
)

criterion = CrossEntropyLoss()


class TransUNetClassification(l.LightningModule):
    def __init__(self, image_size: int, num_classes: int, small=False):
        super().__init__()
        config = CONFIGS["R50-ViT-S_8"] if small else CONFIGS["R50-ViT-B_16"]
        self.network = VisionTransformer(
            config, img_size=image_size, num_classes=num_classes
        )
        self.save_hyperparameters()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        images, classes = batch
        predictions = self(images)
        loss = criterion(predictions, one_hot(classes, num_classes=10).float())
        self.log("training loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, classes = batch
        predictions = self(images)
        accuracy = sum(predictions.topk(1)[1].squeeze() == classes) / len(classes)
        self.log("test accuracy", accuracy, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", "-s", type=bool, default=False)
    args = parser.parse_args()

model = TransUNetClassification(
    image_size=32,
    num_classes=10,
    small=args.small,  # type: ignore
)
logger = WandbLogger(project="TransUNet classification")
trainer = l.Trainer(
    logger=None,
    max_epochs=30,
    enable_progress_bar=True,
    enable_checkpointing=False,
    enable_model_summary=False,
)
trainer.fit(model, train_loader)
trainer.test(model, test_loader)
wandb.finish(quiet=True)
