import torch
import torchvision
from data import Carvana
from torch.utils.data import DataLoader

def save(state, filename = "/content/drive/MyDrive/Carvana/checkpoints/carvana_check16_3_24.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)

def load(chekpoint, model):
    print("Loading checkpoint...")
    model.load_state_dict(chekpoint["state_dict"])

def get_loaders(
        train_folder,
        train_maskfolder,
        test_folder,
        test_maskfolder,
        batch_size,
        train_transform,
        val_transform,
        num_workers = 2,
        pin_memory = True,
):
    train_ds= Carvana(
        images_folder=train_folder,
        masks_folder=train_maskfolder,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_ds = Carvana(
        images_folder=test_folder,
        masks_folder=test_maskfolder,
        transform=val_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers= num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return  train_loader, test_loader

def check_accuracy(loader, model, device="cuda"):
    correct = 0
    total = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds =  (preds > 0.5).float()
            correct += (preds == y).sum()
            total += torch.numel(preds)
            dice_score += (2* (preds * y).sum()) / ((preds + y).sum() + 1e-8)
        print(
            f"Got {correct}/{total} with acc {correct/total*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(loader)}")
        model.train()

        f = open("/content/drive/MyDrive/Carvana/logs/acc/16_3_24_2.txt", "a+")
        f.write(f"{correct/total*100:.2f}\n")
        f.close()

        f = open("/content/drive/MyDrive/Carvana/logs/dice/16_3_24_2.txt", "a+")
        f.write(f"{dice_score/len(loader)}\n")
        f.close()

def save_predictions_as_images(
            loader, model, folder = "/content/drive/MyDrive/Carvana/predictions/", device = "cuda"
    ):
        model.eval()
        print(enumerate(loader))
        for i, (x,y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{i}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{i}.png")

        model.train()
