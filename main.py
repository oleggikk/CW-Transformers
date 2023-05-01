import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
mobilevit_s = torch.load(r"C:\Users\Oleg\Desktop\CW\swin_s3_tiny_224-caltech256-e10-lr0001-t85.pt")

# Предобработка изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root = r'C:\Users\Oleg\Desktop\Caltech256\test'
labels = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label


# Функция, которая принимает путь к изображению, применяет к нему предобработку и передает его в модель для предсказания
def predict(image_path, model):
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        start_time = time.time()
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        end_time = time.time()
    _, predicted = torch.max(output.data, 1)
    return predicted.item(), end_time - start_time

test_img = r'C:\Users\Oleg\Desktop\Caltech256\test\012.binoculars\012_0001.jpg'

pred, time = predict(test_img, mobilevit_s)
pred = id2label[pred]
print(pred, time)