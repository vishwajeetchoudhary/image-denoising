import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from skimage import io, img_as_float
from skimage.transform import resize
import os
import glob

print("Num GPUs Available: ", torch.cuda.device_count())

def radial_crop(image, crop_size, image_size=(400, 600)):
    center_x, center_y = image_size[1] // 2, image_size[0] // 2
    radius = min(center_x, center_y, crop_size // 2)
    mask = torch.zeros((1, image_size[0], image_size[1]), dtype=torch.bool)
    Y, X = torch.meshgrid(torch.arange(image_size[0]), torch.arange(image_size[1]))
    dist_from_center = torch.sqrt((X - center_x)*2 + (Y - center_y)*2)
    mask[dist_from_center <= radius] = True
    cropped_image = torch.zeros_like(image)
    cropped_image[:, mask] = image[:, mask]
    return transforms.ToPILImage()(resize(cropped_image.squeeze(0).permute(1, 2, 0).numpy(), (crop_size, crop_size)))

class ResidualBlock(nn.Module):
    def _init_(self, filters, kernel_size=(3, 3)):
        super(ResidualBlock, self)._init_()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class DenoisingModel(nn.Module):
    def _init_(self):
        super(DenoisingModel, self)._init_()
        self.conv_first = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, padding='same')

    def forward(self, x):
        out = self.relu(self.conv_first(x))
        out = self.res_blocks(out)
        out = self.conv_last(out)
        return x + out

def preprocess_image(img_path, crop_size, image_size=(400, 600)):
    img = img_as_float(io.imread(img_path))
    img = resize(img, image_size)
    img = radial_crop(torch.tensor(img).permute(2, 0, 1).unsqueeze(0), crop_size)
    return transforms.ToTensor()(img)

def psnr(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def train_model(low_folder, high_folder, val_low_folder, val_high_folder, crop_size, input_shape, image_size=(400, 600), epochs=100, batch_size=16, fine_tune=False, model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if fine_tune and model_path:
        model = torch.load(model_path)
    else:
        model = DenoisingModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    def lr_scheduler(epoch):
        return 1e-4 * (0.9 ** (epoch // 10))

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)

    def checkpoint(model):
        torch.save(model.state_dict(), 'denoising_best_model.pth')

    best_psnr = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (low_images, high_images) in enumerate(data_generator(low_folder, high_folder, crop_size, image_size, batch_size)):
            low_images, high_images = low_images.to(device), high_images.to(device)

            optimizer.zero_grad()

            outputs = model(low_images)
            loss = criterion(outputs, high_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for i, (val_low_images, val_high_images) in enumerate(data_generator(val_low_folder, val_high_folder, crop_size, image_size, batch_size)):
                val_low_images, val_high_images = val_low_images.to(device), val_high_images.to(device)

                val_outputs = model(val_low_images)
                val_psnr += psnr(val_high_images, val_outputs).item()

        val_psnr /= len(glob.glob(os.path.join(val_low_folder, '*.png'))) // batch_size

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss}, Val PSNR: {val_psnr}')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint(model)

        lr_scheduler.step()

    return model

def data_generator(low_folder, high_folder, crop_size, image_size=(400, 600), batch_size=32):
    low_image_paths = sorted(glob.glob(os.path.join(low_folder, '*.png')))
    high_image_paths = sorted(glob.glob(os.path.join(high_folder, '*.png')))

    while True:
        for i in range(0, len(low_image_paths), batch_size):
            low_batch_paths = low_image_paths[i:i + batch_size]
            high_batch_paths = high_image_paths[i:i + batch_size]

            low_images = torch.stack([preprocess_image(p, crop_size, image_size) for p in low_batch_paths])
            high_images = torch.stack([preprocess_image(p, crop_size, image_size) for p in high_batch_paths])

            yield low_images, high_images

# Updated folder paths
low_folder = '/pictures/vc/fortrain/low'
high_folder = '/pictures/vc/fortrain/high'
val_low_folder = '/pictures/vc/forval/low'
val_high_folder = '/pictures/vc/forval/high'

low_image_paths = glob.glob(os.path.join(low_folder, '*.png'))
if len(low_image_paths) == 0:
    raise ValueError("No PNG files found in the directory:", low_folder)

crop_size = 256
input_shape = (3, crop_size, crop_size)

model = train_model(low_folder, high_folder, val_low_folder, val_high_folder, crop_size, input_shape, image_size=(400, 600))
torch.save(model.state_dict(), 'denoising1_model.pth')

input_shape_full = (3, 400, 600)
model_fine_tuned = train_model(low_folder, high_folder, val_low_folder, val_high_folder, crop_size, input_shape_full, image_size=(400, 600), fine_tune=True, model_path='denoising_best_model.pth')
torch.save(model_fine_tuned.state_dict(), 'adapted_denoising.pth')
