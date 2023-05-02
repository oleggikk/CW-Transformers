import os
import shutil
from sklearn.model_selection import train_test_split

# Создание папок для тренировочной, валидационной и тестовой выборок
os.makedirs(r'C:\Users\Oleg\Desktop\Caltech256tvt\train', exist_ok=True)
os.makedirs(r'C:\Users\Oleg\Desktop\Caltech256tvt\valid', exist_ok=True)
os.makedirs(r'C:\Users\Oleg\Desktop\Caltech256tvt\test', exist_ok=True)

# Путь к папке с датасетом
path = r"C:\Users\Oleg\Desktop\ObjectCategories"
print(os.path.exists(path))     # True
print(os.listdir(path))

# Разбиваем данные на 3 непересекающиеся части
train_files = []
valid_files = []
test_files = []

for category in os.listdir(path):
    category_path = os.path.join(path, category)
    if os.path.isdir(category_path):
        files = [os.path.join(category_path, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        if len(files) == 0:
            print(category_path)
            raise ValueError("The list of files is empty.")
        train_files_category, test_valid_files_category = train_test_split(files, test_size=0.3, random_state=1)
        valid_files_category, test_files_category = train_test_split(test_valid_files_category, test_size=0.66666, random_state=1)
        train_files.extend(train_files_category)
        valid_files.extend(valid_files_category)
        test_files.extend(test_files_category)

# Копирование файлов в соответствующие папки
for filename in train_files:
    category = os.path.basename(os.path.dirname(filename))
    dst = os.path.join("train", category, os.path.basename(filename))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(filename, dst)

for filename in valid_files:
    category = os.path.basename(os.path.dirname(filename))
    dst = os.path.join("valid", category, os.path.basename(filename))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(filename, dst)

for filename in test_files:
    category = os.path.basename(os.path.dirname(filename))
    dst = os.path.join("test", category, os.path.basename(filename))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(filename, dst)