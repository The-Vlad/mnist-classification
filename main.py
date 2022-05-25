import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Импортируем вспомогательные модули
import matplotlib.pyplot as plt

# Загрузим данные из набора MNIST
train_dataset = datasets.MNIST(root='./data/train', train=True, download=False, transform=ToTensor())
test_dataset = datasets.MNIST(root='./data/test', train=False, download=False, transform=ToTensor())
print('Размер обучающего датасета: ', len(train_dataset))
print('Размер тестирующего датасета: ', len(test_dataset))

# Зададим количество эпох и размер батча
EPOCHS = 5
BATCH_SIZE = 100

# Разобьём выборки на батчи заданного размера
# DataLoader позволяет разбить выборку на пакеты заданного размера.
# Параметр shuffle отвечает за перемешивание данных в пакете
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

for i, (X, y) in enumerate(train_dataloader):
    print(f"Размерность X: Номер батча (итерация) {i} [Размер батча, C?, Высота, Ширина]: {X.shape}")
    print(f"Размерность y: {y.shape} {y.dtype}")
    plt.imshow(X[0][0])
    plt.show()
    break

# Реализуем полносвязную нейронную сеть посредством класса с конструктором и методом forward.
# Класс наследуется от nn.Module
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.flatten = nn.Flatten()
        self.LinearStack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.LinearStack(X)
        return logits

# Если графический ускоритель поддерживает обучение на нем, будем использовать его,
# иначе обучать на процессоре.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Создаём экземпляр модели полносвязной нейронной сети
model = FullyConnectedNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
loss_history = list()  # список для хранения истории изменения функции стоимости
loss_history_log = list() # Логарифмированная история изменения функции стоимости для большей наглядности

# Опишем функцию обучения
def train(model, loss_fn, optimizer, dataloader):
    size = len(dataloader.dataset)
    model.train()
    for batch_iteration, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Рассчитаем прямое распространение
        predicted = model(X)

        # Рассчитаем лосс-функцию
        loss = loss_fn(predicted, y)
        loss_history.append(loss.item())
        loss_history_log.append(loss.log().item())

        # Обратное распространение
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_iteration % 100 == 0:
            loss, current = loss.item(), batch_iteration * len(X)
            print(f"loss: {loss:7f}  [{current:5d}/{size:5d}]")

# Опишем функцию тестирования
def test(model, loss_fn, dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            predicted = model(X)
            test_loss += loss_fn(predicted, y).item()
            correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Точность: {(100*correct):0.1f}%, Среднее значение loss-функции: {test_loss:8f} \n")

# Произведём обучение
for epoch in range(EPOCHS):
    print(f"Эпоха {epoch + 1}\n-------------------------------")
    train(model, loss_fn, optimizer, train_dataloader)
    test(model, loss_fn, test_dataloader)
print("Завершено!")

# Выведем графики изменения функции стоимости со временем обучения
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Изменение функции стоимости со временем обучения')
ax1.plot(loss_history)
ax1.set_title('Обычная история лосс-функции')
ax2.plot(loss_history_log)
ax2.set_title('Логарифмированная история лосс-функции')
plt.show()

# Сохраним обученную модель
torch.save(model.state_dict(), "model.pth")
print("PyTorch Model State сохранено в файле model.pth")
