
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
# Load an image file
image_path = 'image.jpg'  # Replace 'image.jpg' with the path to your image file
image = cv2.imread('/home/fariborz/detectron2/tools/deploy/TRRT_R/left_276439_20230731T122628.png')

# Display the image
plt.imshow(image)
# plt.title('Loaded Image')
# plt.axis('off')  # Hide the axis
plt.show()
