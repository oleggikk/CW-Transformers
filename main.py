from datasets import load_dataset
from datasets import Dataset
from transformers import ViTImageProcessor
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, AdamW
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import pandas as pd
import numpy as np

# Определение устройства для обучения (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

train_path = r"C:\Users\Oleg\Desktop\Caltech256\train"
test_path = r"C:\Users\Oleg\Desktop\Caltech256\test"
val_path = r"C:\Users\Oleg\Desktop\Caltech256\valid"
root_path = r"C:\Users\Oleg\Desktop\Caltech256"

datast = load_dataset(root_path)

train_ds = datast["train"]
test_ds = datast["test"]
val_ds = datast["validation"]

id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}

# Загружаем предобученную модель vit-base-patch16-224-in21k ViTForImageClassification
processor = ViTImageProcessor.from_pretrained("facebook/dino-vits8")
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]
print("mean:", image_mean, "std:", image_std, "size:", size)
# mean, std и size берем из предобученной модели, чтоб трансформировать новые данные

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose([
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,]
    )

_test_val_transforms = Compose([
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def test_val_transforms(examples):
    examples['pixel_values'] = [_test_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

train_ds.set_transform(train_transforms)
val_ds.set_transform(test_val_transforms)
test_ds.set_transform(test_val_transforms)


# Создаем DataLoaders
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_batch_size = 2
eval_batch_size = 2

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)


class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=257):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('facebook/dino-vits8',
                                                             num_labels=257,
                                                             id2label=id2label,
                                                             label2id=label2id)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def test_dataloader(self):
        return test_dataloader


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='/content/drive/MyDrive/Caltech256/models',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

model = ViTLightningModule()
trainer = Trainer(accelerator = "gpu", callbacks=[EarlyStopping(monitor='validation_loss'), checkpoint_callback], max_epochs=10)
trainer.fit(model, train_dataloader, val_dataloader)

