import torch
import torchvision.transforms as transforms
import timm
import os
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

# Переводим вычисления на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Для перевода id в соответствующий класс
root_path = r'C:\Users\Oleg\Desktop\Caltech256\test'
labels = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# Загружаем модель
model = torch.load(r'C:\Users\Oleg\Desktop\CW\models\deit_tiny_distilled_patch16_224-caltech256-e10-lr0001-t79.pt')
model.eval()

# Предобработка изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def prepareImage(image):
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    return image_tensor

# Функция передает тенсор изображения в модель для предсказания
def predict(image_tensor, model):
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

test_img = r'C:\Users\Oleg\Desktop\Caltech256\test\012.binoculars\012_0001.jpg'


# Функция для изменения размера изображения
def resize_image(image):
    # Получаем текущий размер изображения
    width, height = image.size

    # Вычисляем соотношение сторон
    ratio = min(500 / width, 500 / height)

    # Вычисляем новый размер изображения
    new_size = (int(width * ratio), int(height * ratio))

    # Изменяем размер изображения
    resized_image = image.resize(new_size)

    return resized_image

# Функция для выбора изображения
def choose_image():
    # Открываем диалоговое окно для выбора файла
    file_path = filedialog.askopenfilename()

    # Открываем изображение
    image = Image.open(file_path)

    # Изменяем размер изображения
    resized_image = resize_image(image)

    # Создаем объект PhotoImage для отображения изображения в окне
    image_tk = ImageTk.PhotoImage(resized_image)

    # Обновляем виджет Label для отображения нового изображения
    label.configure(image=image_tk)
    label.image = image_tk

    # Обновляем подпись с путем к файлу
    file_label.configure(text="Выбранный файл: " + file_path)

    # Готовим тензор для подачи в модель
    image_tensor = prepareImage(image)

    # Передаем объект в модель
    pred = predict(image_tensor, model)
    pred = id2label[pred]

    # Выводим предсказанный класс модели
    prediction.configure(text= "predicted class: " + pred)

root = Tk()
root.geometry("600x600")
root.title("Выберите изображение")

# Создаем кнопку для повторного выбора изображения
button = Button(root, text="Выбрать изображение", command=choose_image)
button.pack()

# Создаем начальную подпись с инструкцией
initial_label = Label(root, text="Нажмите на кнопку, чтобы выбрать изображение")
initial_label.pack()

# Создаем пустой виджет Label для отображения изображения
label = Label(root)
label.pack()

# Создаем пустую подпись для отображения пути к файлу
file_label = Label(root)
file_label.pack()

# Создаем пустую подпись для отображения
prediction = Label()
prediction.pack()

root.mainloop()

