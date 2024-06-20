import torch
import torch.nn as nn
import torchvision.transforms as transforms
from skimage import io, img_as_float
from skimage.transform import resize
import os
import glob

class DenoisingModel(nn.Module):
    def _init_(self):
        super(DenoisingModel, self)._init_()
        # Define your model layers here
        self.conv1 = nn.Conv2d(...)
        self.relu = nn.ReLU()
        # Define more layers as needed
    
    def forward(self, x):
        # Define the forward pass of your model
        x = self.conv1(x)
        x = self.relu(x)
        # Add more layers as needed
        return x

def preprocess_image(img_path, image_size=(400, 600)):
    img = img_as_float(io.imread(img_path))
    img = resize(img, image_size)
    img = transforms.ToTensor()(img)
    return img

def save_image(image, path):
    image = torch.clamp(image, 0, 1)
    image = (image * 255).cpu().numpy().astype(np.uint8)
    io.imsave(path, image.transpose(1, 2, 0))

model_path = 'adapted_denoising.pth'
model = DenoisingModel()
model.load_state_dict(torch.load(model_path))

test_low_folder = '/test/low'
test_pred_folder = '/test/predicted'

os.makedirs(test_pred_folder, exist_ok=True)

test_image_paths = sorted(glob.glob(os.path.join(test_low_folder, '*.png')))
for img_path in test_image_paths:
    img = preprocess_image(img_path)
    img = img.unsqueeze(0)

    with torch.no_grad():
        denoised_img = model(img)

    base_name = os.path.basename(img_path)
    save_image(denoised_img[0], os.path.join(test_pred_folder, base_name))

print("Denoised images saved to:", test_pred_folder)