import cv2
import torch

def preprocess(img):
    """Resize the image to 84x84 and convert it from color to grayscale"""
    img = img[:84, 6:90] 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0

    for i in range(84):
        for j in range(84):
            if 65<=i<=77 and 37<=j<=45:  #car
                img[i,j]=0
            elif 0.38<=img[i,j]<=0.42:  #street
                img[i,j]=0.4
            elif img[i,j]<0.6565: #dark grass
                img[i,j]=0.627
            else: #light grass
                img[i,j]=0.686

    return img

def postprocess(images):
    # Assume 'images' is a tensor of shape [batch_size, 4,84, 84]

    # Create a new tensor for processed images, initially copying all values from images
    processed_images = images.clone()

    # Define the region for the 'car'
    car_mask = torch.zeros_like(images, dtype=torch.bool)
    car_mask[:,65:78, 37:46] = True  # using slicing to set the car region to True
    processed_images[car_mask] = 0  # set the car region to 0

    # Define the thresholds for 'street'
    street_mask = (images > 0.3) & (images <= 0.55)
    processed_images[street_mask] = 0.4

    # Define the threshold for 'dark grass'
    dark_grass_mask = (images < 0.6565) & (images>0.55) 
    processed_images[dark_grass_mask] = 0.627


    # Define the threshold for 'light grass'
    # This is essentially the else condition in the original loop
    light_grass_mask = ~car_mask & ~street_mask & ~dark_grass_mask
    processed_images[light_grass_mask] = 0.686


    return processed_images