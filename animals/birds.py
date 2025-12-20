import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

classNames = ['Гаички', 'Домовый воробей', 'Красный кардинал', 'Кряква', 'Обыкновенный скворец']

testTransform = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cpu")

class BirdCNN(nn.Module):
    def __init__(self, numClasses=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, numClasses)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def loadModel(modelName):
    print()
    modelPath = os.path.join(os.getcwd(), f'animals/models/best_{modelName}.pth')
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Модель не найдена: {modelPath}")

    if modelName == "BirdCNN":
        model = BirdCNN(numClasses=5)
    elif modelName == "ResNet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 5)
    elif modelName == "EfficientNetB0":
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    elif modelName == "DenseNet121":
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, 5)
    elif modelName == "MobileNetV3_Small":
        model = models.mobilenet_v3_small()
        numFeatures = model.classifier[0].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
        model.classifier = nn.Sequential(nn.Linear(numFeatures, 5))
    else:
        raise ValueError(f"Неизвестная модель: {modelName}")

    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.to(device)
    model.eval()
    return model


class BirdClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Классификатор Птиц")
        self.resize(1000, 600)
        self.setMinimumSize(500, 300)

        mainLayout = QHBoxLayout()

        controlPanel = QVBoxLayout()
        controlPanel.setAlignment(Qt.AlignTop)
        
        self.modelCombo = QComboBox()
        self.modelCombo.addItems(["BirdCNN", "ResNet18", "EfficientNetB0", "DenseNet121", "MobileNetV3_Small"])
        controlPanel.addWidget(self.modelCombo)

        self.loadButton = QPushButton("Загрузить изображение")
        self.loadButton.clicked.connect(self.loadImage)
        controlPanel.addWidget(self.loadButton)

        self.classifyButton = QPushButton("Классифицировать")
        self.classifyButton.clicked.connect(self.classifyImage)
        controlPanel.addWidget(self.classifyButton)

        self.resultLabel = QLabel("Результат: ")
        self.resultLabel.setWordWrap(True)
        controlPanel.addWidget(self.resultLabel)

        controlPanel.addStretch()

        controlWidget = QWidget()
        controlWidget.setLayout(controlPanel)
        controlWidget.setFixedWidth(180)
        mainLayout.addWidget(controlWidget)

        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setStyleSheet("border: 2px solid #555; background-color: #f0f0f0;")
        self.imageLabel.setScaledContents(True)
        mainLayout.addWidget(self.imageLabel)

        self.setLayout(mainLayout)

        self.imagePath = None

    def loadImage(self):
        fileDialog = QFileDialog(self)
        imagesPath = os.path.join(os.getcwd(), f'animals/datasets/birds')
        self.imagePath, _ = fileDialog.getOpenFileName(
            self, "Выберите изображение", imagesPath, "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if self.imagePath:
            pixmap = QPixmap(self.imagePath)
            scaledPixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.imageLabel.setPixmap(scaledPixmap)
            self.resultLabel.setText("Изображение загружено.\nВыберите модель и нажмите 'Классифицировать'.")

    def classifyImage(self):
        if not self.imagePath:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение!")
            return

        modelName = self.modelCombo.currentText()
        try:
            model = loadModel(modelName)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{e}")
            return

        try:
            image = Image.open(self.imagePath).convert("RGB")
            inputTensor = testTransform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(inputTensor)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                predIdx = torch.argmax(probs).item()
                predClass = classNames[predIdx]
                confidence = probs[predIdx].item() * 100

            self.resultLabel.setText(
                f"<b>Класс:</b> {predClass}<br>"
                f"<b>Вероятность:</b> {confidence:.2f}%"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при классификации:\n{e}")


if __name__ == "__main__":
    print(os.getcwd())
    app = QApplication(sys.argv)
    window = BirdClassifier()
    window.show()
    sys.exit(app.exec_())