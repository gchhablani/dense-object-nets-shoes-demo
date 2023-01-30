import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from .constants import CUSTOM_IMAGES_MULTI_DIR, CUSTOM_IMAGES_SINGLE_DIR

def get_directories_and_files(logs_proto_path, scene_name):
    image_masks_dir = os.path.join(logs_proto_path, scene_name, "processed/image_masks")
    images_dir = os.path.join(logs_proto_path, scene_name, "processed/images")
    rendered_images_dir = os.path.join(logs_proto_path, scene_name, "processed/rendered_images")
    fusion_mesh_file = os.path.join(logs_proto_path, scene_name, "processed/fusion_mesh.ply")
    fusion_pointcloud_file = os.path.join(logs_proto_path, scene_name, "processed/fusion_pointcloud.ply")

    return images_dir, image_masks_dir, rendered_images_dir, fusion_mesh_file, fusion_pointcloud_file

def get_images_given_rbg_image_file(logs_proto_path, scene_name, rgb_image_file):
    images_dir, image_masks_dir, rendered_images_dir, fusion_mesh_file, fusion_pointcloud_file = get_directories_and_files(logs_proto_path, scene_name)
    image_idx = rgb_image_file.split('.')[0][:-4]
    rgb_file = os.path.join(images_dir, f"{image_idx}_rgb.png")
    rgb = plt.imread(rgb_file)

    depth_file = os.path.join(rendered_images_dir, f"{image_idx}_depth.png")
    depth = plt.imread(depth_file)

    depth_cropped_file = os.path.join(rendered_images_dir, f"{image_idx}_depth_cropped.png")
    depth_cropped = plt.imread(depth_cropped_file)

    mask_file = os.path.join(image_masks_dir, f"{image_idx}_mask.png")
    mask = plt.imread(mask_file)


    return rgb, depth, mask, depth_cropped, fusion_pointcloud_file

def get_tensor_from_numpy_image(numpy_image, transform):
    transformed_image = transform(numpy_image)
    return transformed_image.unsqueeze(0)
    
def get_numpy_image_from_tensor_descriptor(output):
    output_descriptor = output.squeeze(0).permute(1, 2, 0).detach().numpy()
    return output_descriptor

def get_random_image_file(logs_root_path, scene_name):
    scene_directory = os.path.join(logs_root_path, scene_name, 'processed')
    rgb_images_regex = os.path.join(scene_directory, "images/*_rgb.png")
    all_rgb_images_in_scene = glob.glob(rgb_images_regex)
    image_file = random.choice(all_rgb_images_in_scene).split('/')[-1]
    return image_file

def get_random_scene_name(logs_root_path):
    scenes = os.listdir(logs_root_path)
    return random.choice(scenes)

def get_random_pixel_from_mask_image(mask_image):
    return random.choice(np.transpose(np.nonzero(mask_image)))

def get_random_custom_image():
    all_rgb_images_in_scene = os.listdir(CUSTOM_IMAGES_SINGLE_DIR)
    image_file = os.path.join(CUSTOM_IMAGES_SINGLE_DIR, random.choice(all_rgb_images_in_scene))
    image = plt.imread(image_file)
    resized_image = cv2.resize(image, (640, 480))
    return resized_image

def get_random_custom_image_multi():
    all_rgb_images_in_scene = os.listdir(CUSTOM_IMAGES_MULTI_DIR)
    image_file = os.path.join(CUSTOM_IMAGES_MULTI_DIR, random.choice(all_rgb_images_in_scene))
    image = plt.imread(image_file)
    resized_image = cv2.resize(image, (640, 480))
    return resized_image