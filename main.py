import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.dataset import get_dataloaders
from src.model import get_resnet18_model
from src.trainer import train_model

def save_plots(history, filename='loss_acc_graph.png'):
    """
    하나의 캔버스(Figure) 안에 두 개의 서브플롯(Subplot)을 생성하여
    왼쪽에는 Accuracy, 오른쪽에는 Loss 그래프를 그립니다.
    """
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    epochs = range(1, len(train_acc) + 1)

    # 1행 2열의 구조로 그래프 생성 (가로 길이 14, 세로 길이 6)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------------------------
    # 첫 번째 그래프: Accuracy
    # ---------------------------
    axes[0].plot(epochs, train_acc, 'bo-', label='Training Acc')
    axes[0].plot(epochs, val_acc, 'ro-', label='Validation Acc')
    axes[0].set_title('Training and Validation Accuracy', fontsize=15)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # ---------------------------
    # 두 번째 그래프: Loss
    # ---------------------------
    axes[1].plot(epochs, train_loss, 'bo-', label='Training Loss')
    axes[1].plot(epochs, val_loss, 'ro-', label='Validation Loss')
    axes[1].set_title('Training and Validation Loss', fontsize=15)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    # 레이아웃 간격 자동 조절
    plt.tight_layout()
    
    # 파일 저장
    plt.savefig(filename)
    print(f"📊 결과 그래프가 '{filename}' 파일로 저장되었습니다.")
    plt.close()

def main():
    # 1. 설정 (Configuration)
    DATA_PATH = 'catanddog' 
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Using Device: {DEVICE}")

    # 2. 데이터 로드
    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_PATH, BATCH_SIZE)
    print(f"Classes: {class_names}")

    # 3. 모델 준비
    model = get_resnet18_model(num_classes=len(class_names), pretrained=True, freeze_backbone=True)
    model = model.to(DEVICE)

    # 4. 손실함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

    # 5. 학습 시작
    print("학습을 시작합니다...")
    model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, DEVICE, NUM_EPOCHS)

    # 6. 결과 그래프 저장
    save_plots(history, 'loss_acc_graph.png')

    # 7. 모델 저장
    save_path = 'resnet18_catdog.pth'
    torch.save(model.state_dict(), save_path)
    print(f"💾 모델이 '{save_path}' 경로에 저장되었습니다.")

if __name__ == '__main__':
    main()