from torchvision import transforms
from constants import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD_DEV

image_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD_DEV)])

