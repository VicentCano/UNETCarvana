import torch
import albumentations
from albumentations.pytorch import ToTensorV2
from tqdm import  tqdm
import  torch.nn as NeuralNet
import torch.optim as to
from unet import UNET
from extra import(
    load,
    save,
    get_loaders,
    check_accuracy,
    save_predictions_as_images
)

# Hyperparameters
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 4
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMAGES = "/content/drive/MyDrive/Carvana/train/"
TRAIN_MASKS = "/content/drive/MyDrive/Carvana/train_masks/"
TEST_IMAGES = "/content/drive/MyDrive/Carvana/test_subset"
TEST_MASKS = "/content/drive/MyDrive/Carvana/mask_subset"

def train(loader, model, optimizer, loss_function, scaler):
    loop = tqdm(loader)

    for batch_i, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_function(predictions,targets)
        
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())


def main():
    train_transform = albumentations.Compose(
        [
            albumentations.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            albumentations.Rotate(limit=35,p=1.0),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.1),
            albumentations.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


    val_transforms = albumentations.Compose(
       [
           albumentations.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
           albumentations.Normalize(
               mean=[0.0, 0.0, 0.0],
               std=[1.0, 1.0, 1.0],
               max_pixel_value=255.0,
           ),
           ToTensorV2(),
       ],
   )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_function = NeuralNet.BCEWithLogitsLoss()
    optimizer = to.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMAGES,
        TRAIN_MASKS,
        TEST_IMAGES,
        TEST_MASKS,
        BATCH_SIZE,
        train_transform,
        val_transforms
    )

    if LOAD_MODEL:
        load(torch.load("/content/drive/MyDrive/Carvana/checkpoints/carvana_check16_3_24.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train(train_loader,model,optimizer,loss_function, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_images(
            val_loader, model, folder="/content/drive/MyDrive/Carvana/predictions/", device=DEVICE
        )
    
if __name__ == "__main__":
    main()

