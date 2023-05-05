import torch
import torchvision.transforms as transforms
import timm
import os
from tkinter import filedialog
from tkinter import *
from tkinter.ttk import Combobox
from PIL import ImageTk, Image
#import torchvision
#print(torch.__version__)
#print(torchvision.__version__)
#print(timm.__version__)

# получаем путь к директории, в которой находится запущенный скрипт
script_dir = os.path.dirname(__file__)

# Переводим вычисления на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Для перевода id в соответствующий класс
labels = ['001.ak47', '002.american-flag', '003.backpack', '004.baseball-bat', '005.baseball-glove', '006.basketball-hoop',
          '007.bat', '008.bathtub', '009.bear', '010.beer-mug', '011.billiards', '012.binoculars', '013.birdbath', '014.blimp',
          '015.bonsai-101', '016.boom-box', '017.bowling-ball', '018.bowling-pin', '019.boxing-glove', '020.brain-101', '021.breadmaker',
          '022.buddha-101', '023.bulldozer', '024.butterfly', '025.cactus', '026.cake', '027.calculator', '028.camel', '029.cannon', '030.canoe',
          '031.car-tire', '032.cartman', '033.cd', '034.centipede', '035.cereal-box', '036.chandelier-101', '037.chess-board', '038.chimp',
          '039.chopsticks', '040.cockroach', '041.coffee-mug', '042.coffin', '043.coin', '044.comet', '045.computer-keyboard', '046.computer-monitor',
          '047.computer-mouse', '048.conch', '049.cormorant', '050.covered-wagon', '051.cowboy-hat', '052.crab-101', '053.desk-globe', '054.diamond-ring',
          '055.dice', '056.dog', '057.dolphin-101', '058.doorknob', '059.drinking-straw', '060.duck', '061.dumb-bell', '062.eiffel-tower', '063.electric-guitar-101',
          '064.elephant-101', '065.elk', '066.ewer-101', '067.eyeglasses', '068.fern', '069.fighter-jet', '070.fire-extinguisher', '071.fire-hydrant',
          '072.fire-truck', '073.fireworks', '074.flashlight', '075.floppy-disk', '076.football-helmet', '077.french-horn', '078.fried-egg', '079.frisbee',
          '080.frog', '081.frying-pan', '082.galaxy', '083.gas-pump', '084.giraffe', '085.goat', '086.golden-gate-bridge', '087.goldfish', '088.golf-ball',
          '089.goose', '090.gorilla', '091.grand-piano-101', '092.grapes', '093.grasshopper', '094.guitar-pick', '095.hamburger', '096.hammock', '097.harmonica',
          '098.harp', '099.harpsichord', '100.hawksbill-101', '101.head-phones', '102.helicopter-101', '103.hibiscus', '104.homer-simpson', '105.horse', '106.horseshoe-crab',
          '107.hot-air-balloon', '108.hot-dog', '109.hot-tub', '110.hourglass', '111.house-fly', '112.human-skeleton', '113.hummingbird', '114.ibis-101', '115.ice-cream-cone',
          '116.iguana', '117.ipod', '118.iris', '119.jesus-christ', '120.joy-stick', '121.kangaroo-101', '122.kayak', '123.ketch-101', '124.killer-whale', '125.knife', '126.ladder',
          '127.laptop-101', '128.lathe', '129.leopards-101', '130.license-plate', '131.lightbulb', '132.light-house', '133.lightning', '134.llama-101', '135.mailbox', '136.mandolin',
          '137.mars', '138.mattress', '139.megaphone', '140.menorah-101', '141.microscope', '142.microwave', '143.minaret', '144.minotaur', '145.motorbikes-101', '146.mountain-bike',
          '147.mushroom', '148.mussels', '149.necktie', '150.octopus', '151.ostrich', '152.owl', '153.palm-pilot', '154.palm-tree', '155.paperclip', '156.paper-shredder', '157.pci-card',
          '158.penguin', '159.people', '160.pez-dispenser', '161.photocopier', '162.picnic-table', '163.playing-card', '164.porcupine', '165.pram', '166.praying-mantis', '167.pyramid',
          '168.raccoon', '169.radio-telescope', '170.rainbow', '171.refrigerator', '172.revolver-101', '173.rifle', '174.rotary-phone', '175.roulette-wheel', '176.saddle', '177.saturn',
          '178.school-bus', '179.scorpion-101', '180.screwdriver', '181.segway', '182.self-propelled-lawn-mower', '183.sextant', '184.sheet-music', '185.skateboard', '186.skunk', '187.skyscraper',
          '188.smokestack', '189.snail', '190.snake', '191.sneaker', '192.snowmobile', '193.soccer-ball', '194.socks', '195.soda-can', '196.spaghetti', '197.speed-boat', '198.spider', '199.spoon',
          '200.stained-glass', '201.starfish-101', '202.steering-wheel', '203.stirrups', '204.sunflower-101', '205.superman', '206.sushi', '207.swan', '208.swiss-army-knife', '209.sword', '210.syringe',
          '211.tambourine', '212.teapot', '213.teddy-bear', '214.teepee', '215.telephone-box', '216.tennis-ball', '217.tennis-court', '218.tennis-racket', '219.theodolite', '220.toaster', '221.tomato',
          '222.tombstone', '223.top-hat', '224.touring-bike', '225.tower-pisa', '226.traffic-light', '227.treadmill', '228.triceratops', '229.tricycle', '230.trilobite-101', '231.tripod', '232.t-shirt',
          '233.tuning-fork', '234.tweezer', '235.umbrella-101', '236.unicorn', '237.vcr', '238.video-projector', '239.washing-machine', '240.watch-101', '241.waterfall', '242.watermelon', '243.welding-mask',
          '244.wheelbarrow', '245.windmill', '246.wine-bottle', '247.xylophone', '248.yarmulke', '249.yo-yo', '250.zebra', '251.airplanes-101', '252.car-side-101', '253.faces-easy-101',
          '254.greyhound', '255.tennis-shoes', '256.toad', '257.clutter']
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# Загружаем модель
model = torch.load(script_dir + r'\deit_tiny_distilled_patch16_224-caltech256.pt')
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

# Функция для выбора нейросетевой модели
def choose_model():
    # Создаем окно для выбора модели
    model_window = Toplevel(root)
    model_window.title("Выбрать модель")
    model_window.geometry("300x130")

    # Определяем доступные модели
    models = ['deit_tiny_distilled_patch16_224-caltech256.pt', 'eficcientformer_l1-caltech256.pt', 'mobilevits-caltech256.pt',
              'swin_s3_tiny_224-caltech256.pt', 'swin_tiny_patch4_window7_224-caltech256.pt', 'vit_small_patch16_224-caltech256.pt']

    # Создаем выпадающий список
    model_combobox = Combobox(model_window, values=models, state='readonly')
    model_combobox.current(0)
    model_combobox.pack()

    # Функция для закрытия окна выбора модели и возврата выбранной модели
    def close_window():
        chosen_model = model_combobox.get()
        model_window.destroy()
        global model
        model = torch.load(script_dir + '\\' + chosen_model)
        model.eval()
        model_name .configure(text="Выбранная модель: " + chosen_model)
        return chosen_model

    # Создаем кнопку "Ок"
    ok_button = Button(model_window, text="Ок", command=close_window)
    ok_button.pack()

    # Ожидаем закрытия окна выбора модели
    root.wait_window(model_window)

    return close_window()

root = Tk()
root.geometry("600x700")
root.title("Predict Caltech256 class")

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

# Создаем кнопку для выбора нейросетевой модели
model_button = Button(root, text="Выбрать нейросетевую модель", command=choose_model)
model_button.pack()

model_name = Label(text="Выбранная модель: " + "deit_tiny_distilled_patch16_224-caltech256.pt")
model_name.pack()

root.mainloop()


