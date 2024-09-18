import os.path

import cv2
folder_path = '/home/fariborz/Downloads/Data_Z_Camera_Multi_Class_august_23/Data_Z_Camera_Multi_Class/z_camera_data_updated_name'
save_address = '/home/fariborz/Downloads/resized_imagecon'
# Read the image
# input_image = cv2.imread("left_1680715422944095089.jpg")  # Replace "input.jpg" with your image file's path
#
# # Define the desired dimensions for resizing
# new_width = 1008
# new_height = 624
image_path = '//home/fariborz/detectron2/tools/deploy/sampleor.jpg'
filename ='resized_pick_march.jpg'
im = cv2.imread(image_path)
if im is not None:
    if im.shape[0] > 0 and im.shape[1] > 0:
        im = cv2.resize(im, (1280, 720), interpolation=cv2.INTER_CUBIC)
        add = os.path.join(save_address, filename)
        cv2.imwrite(add, im)
        stop = 0



for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # filename ='left_1680712522965841424.jpg'
        print(filename)
        # Read the image using OpenCV
        # filename ='left_1680801979379691428.jpg'
        image_path = os.path.join(folder_path, filename)
        # image_path ='img/crop_b.jpg'
        image_path = '/home/fariborz/detectron2/tools/deploy/sample.jpg'
        im = cv2.imread(image_path)
        if im is not None:
            if im.shape[0] > 0 and im.shape[1] > 0:
                  im = cv2.resize(im, (960, 540), interpolation=cv2.INTER_CUBIC)
                  add= os.path.join(save_address, filename)
                  cv2.imwrite(add, im)
# Resize the image
#resized_image = cv2.resize(input_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Save the resized image
#cv2.imwrite("resized_image0.jpg", resized_image)  # Replace "resized_image.jpg" with your desired output path
