import sys
sys.path.append('/homes/frinaldi/Merge2Update')
from src.dataset.eurosat import EuroSat
import timm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torch import nn, optim
from pathlib import Path

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
    return running_loss / len(train_loader.dataset)

def validate(model, val_loader, device):
    model.eval()
    correct = 0 
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
            
def main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Batch size")
    parser.add_argument("--epochs", type=float, default=10, help="Number of epochs")
    parser.add_argument("--ckpt_save_path", type=str, default=".", help="Number of epochs")
    args = parser.parse_args()   
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    # Define dataset transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = EuroSat(root='/work/debiasing/frinaldi/mammoth/eurosat', split='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataset = EuroSat(root='/work/debiasing/frinaldi/mammoth/eurosat', split='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 10)
    model.to(device)
    
    optimizer= optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = args.epochs
    best_accuracy = 0
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        # Evaluate model
        val_accuracy = validate(model, test_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(),  Path(args.ckpt_save_path, f"best.pt"))
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")
    
if __name__ == "__main__":
    main()