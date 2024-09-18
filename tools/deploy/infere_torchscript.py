# import torch
# import numpy as np
# from PIL import Image
# import torchvision
# import json
# import matplotlib.pyplot as plt
# import cv2
#
# # with open('class_mapping.json') as data:
# #     mappings = json.load(data)
# #
# # class_mapping = {item['model_idx']: item['class_name'] for item in mappings}
# class_mapping = {0:'package'}
# y =  {}
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# model = torch.jit.load('torch_script_output/model.ts').to(device)
#
# image_path = 'crop_b.jpg'
# image = Image.open(image_path)
# # Transform your image if the config.yaml shows
# # you used any image transforms for validation data
# image = np.array(image)
# h, w = image.shape[:2]
# # Convert to torch tensor
# x = torch.from_numpy(image).to(device)
# with torch.no_grad():
#     # Convert to channels first, convert to float datatype
#     x = x.permute(2, 0, 1).float()
#     y_out = model(x)
#     # Some optional postprocessing, you can change the 0.5 iou
#     # overlap as needed
#     y['pred_boxes'] = y_out[0]
#     y['pred_classes'] = y_out [1]
#     y['pred_masks'] = y_out[2]
#     y['scores'] = y_out[3]
#     to_keep = torchvision.ops.nms(y['pred_boxes'], y['scores'], 0.5)
#     y['pred_boxes'] = y['pred_boxes'][to_keep]
#     y['pred_classes'] = y['pred_classes'][to_keep]
#     y['pred_masks'] = y['pred_masks'][to_keep]
#
#     # Draw you box predictions:
#     all_masks = np.zeros((h, w), dtype=np.int8)
#     instance_idx = 1
#     for mask, bbox, label in zip(reversed(y['pred_masks']),
#                                  y['pred_boxes'],
#                                  y['pred_classes']):
#         mask=mask.cpu().numpy(); bbox= bbox.cpu().numpy();label=label.cpu().numpy()
#
#         bbox = list(map(int, bbox))
#         x1, y1, x2, y2 = bbox
#         class_idx = label.item()
#         class_name = class_mapping[class_idx]
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
#         cv2.putText(
#             image,
#             class_name,
#             (x1, y1),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             4,
#             (255, 0, 0)
#         )
#         all_masks[mask[0] == 1] = instance_idx
#
#         instance_idx += 1
# # Display predicted masks, boxes and classes on your image
# plt.imshow(image)
# plt.imshow(all_masks, alpha=0.5)
# plt.show()

import torch
import numpy as np
from PIL import Image
import torchvision
import json
import matplotlib.pyplot as plt
import cv2
import time
import os
import cv2
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

def paste_mask_in_image_old(mask, box, img_h, img_w, threshold):
    """
    Paste a single mask in an image.
    This is a per-box implementation of :func:`paste_masks_in_image`.
    This function has larger quantization error due to incorrect pixel
    modeling and is not used any more.

    Args:
        mask (Tensor): A tensor of shape (Hmask, Wmask) storing the mask of a single
            object instance. Values are in [0, 1].
        box (Tensor): A tensor of shape (4, ) storing the x0, y0, x1, y1 box corners
            of the object instance.
        img_h, img_w (int): Image height and width.
        threshold (float): Mask binarization threshold in [0, 1].

    Returns:
        im_mask (Tensor):
            The resized and binarized object mask pasted into the original
            image plane (a tensor of shape (img_h, img_w)).
    """
    # Conversion from continuous box coordinates to discrete pixel coordinates
    # via truncation (cast to int32). This determines which pixels to paste the
    # mask onto.
    box = box.to(dtype=torch.int32)  # Continuous to discrete coordinate conversion
    # An example (1D) box with continuous coordinates (x0=0.7, x1=4.3) will map to
    # a discrete coordinates (x0=0, x1=4). Note that box is mapped to 5 = x1 - x0 + 1
    # pixels (not x1 - x0 pixels).
    samples_w = box[2] - box[0] + 1  # Number of pixel samples, *not* geometric width
    samples_h = box[3] - box[1] + 1  # Number of pixel samples, *not* geometric height

    # Resample the mask from it's original grid to the new samples_w x samples_h grid
    mask = Image.fromarray(mask.cpu().numpy())
    mask = mask.resize((samples_w, samples_h), resample=Image.BILINEAR)
    mask = np.array(mask, copy=False)

    if threshold >= 0:
        mask = np.array(mask > threshold, dtype=np.uint8)
        mask = torch.from_numpy(mask)
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask = torch.from_numpy(mask * 255).to(torch.uint8)

    im_mask = torch.zeros((img_h, img_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, img_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, img_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


# with open('path_to_the_mappings') as data:
#     mappings = json.load(data)
#
# class_mapping = {item['model_idx']: item['class_name'] for item in mappings}
class_mapping = {0:'box', 1: 'flat', 2: 'polybag', 3: 'others'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = torch.jit.load("/home/fariborz/detectron2/tools/deploy/output/model.ts").to(device)
folder_path ='/home/fariborz/detectron2/tools/deploy/train_200_batch/images_200'

address_ = '/home/fariborz/detectron2/tools/deploy/train_200_batch'
register_coco_instances("experiment", {}, os.path.join(address_, "batch_200.json"),
                        os.path.join(address_, "images_200"))

metadata = MetadataCatalog.get("experiment")
dataset_dicts = DatasetCatalog.get("experiment")


# Initialize the Visualizer
file_name_no_pre = {}
file_name_no_nms = {}

file_name_nms_sf = {}

for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        #filename ='202307261438071690396717_target_rgb_left.jpg'
        print(filename)
        # Read the image using OpenCV
        # filename ='left_1680801979379691428.jpg'
        image_path = os.path.join(folder_path, filename)
        # image_path ='img/crop_b.jpg'
        im = cv2.imread(image_path)
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_CUBIC)
        v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
        # plt.imshow(image)
        # plt.show()
        if im is not None:
            # if im.shape[0] > 0 and im.shape[1] > 0:
            #     image= cv2.resize(im, (1008, 624), interpolation=cv2.INTER_CUBIC)

            #image = Image.open(image_path).convert('RGB')
            # Transform your image if the config.yaml shows
            # you used any image transforms for validation data
            image = np.array(image)
            h, w = image.shape[:2]
            # Convert to torch tensor
            x = torch.from_numpy(image).to(device)

            predicted_instances = []
            with torch.no_grad():
                # Convert to channels first, convert to float datatype
                x = x.permute(2, 0, 1).float()

                start = time.time()
                pred_boxes, pred_classes, pred_masks, scores, _ = model(x)
                print(scores)
                print(len(pred_boxes))
                file_name_no_pre[filename] = len(pred_boxes)
                end = time.time()
                print(end-start)
                score_threshold=0.5
                keep_ = scores > score_threshold
                # print(keep_)
                pred_boxes = pred_boxes[keep_]
                pred_masks = pred_masks[keep_]
                scores = scores[keep_]
                file_name_no_nms[filename] = len(pred_boxes)
                # print(len(scores))
                # Some optional postprocessing, you can change the 0.5 iou
                # overlap as needed
                # to_keep = torchvision.ops.nms(pred_boxes, scores, 0.4)
                # pred_boxes = pred_boxes[to_keep]
                # pred_classes = pred_classes[to_keep]
                # pred_masks = pred_masks[to_keep]
                # print(print(len(pred_masks)))
                file_name_nms_sf[filename] = len(pred_boxes)


                pred_masks_postprocessed = []
                for box, mask in zip(pred_boxes, pred_masks):
                    pred_masks_postprocessed.append(paste_mask_in_image_old(mask[0], box, h, w, 0.5))

                # Draw you box predictions:
                all_masks = np.zeros((h, w), dtype=np.int8)
                instance_idx = 1
                ins_pred_masks=[];ins_pred_boxes=[]
                for mask, bbox, label in zip(pred_masks_postprocessed,
                                             pred_boxes,
                                             pred_classes):
                    bbox = list(map(int, bbox))
                    x1, y1, x2, y2 = bbox
                    class_idx = label.item()
                    class_name = class_mapping[class_idx]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
                    cv2.putText(
                        image,
                        class_name,
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0)
                    )
                    mask = cv2.resize(mask.squeeze().numpy(), dsize=(w, h),
                                      interpolation=cv2.INTER_LINEAR)
                    # plt.imshow(mask)
                    # plt.show()

                    all_masks[mask > 0] = instance_idx
                    ins_pred_masks.append(mask)
                    ins_pred_boxes.append([x1, y1, x2, y2])
                instance = Instances(image_size=(image.shape[0], image.shape[1]))
                instance.pred_masks = ins_pred_masks
                instance.pred_boxes = ins_pred_boxes

                    # instance_idx += 1
            # Display predicted masks, boxes and classes on your image
            v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
            v=v.draw_instance_predictions(instance.to("cpu"))

            # v = v.draw_instance_predictions(predicted_instances[0])

            plt.imshow(v.get_image())
            plt.show()

output_file_file_name_no_pre = "file_name_no_pre_torch.txt"
            # Open the output file in write mode
with open(output_file_file_name_no_pre , "w") as file:
                # Iterate over the file data and write each entry to the file
    for file_name, number in file_name_no_pre.items():
            file.write(f"{file_name}: {number}\n")

output_file_file_name_no_nms = "file_name_no_nms_torch.txt"
            # Open the output file in write mode
with open(output_file_file_name_no_nms, "w") as file:
                # Iterate over the file data and write each entry to the file
    for file_name, number in file_name_no_nms.items():
            file.write(f"{file_name}: {number}\n")

output_file_file_name_nms_sf = "file_name_nms_sf_torch.txt"
            # Open the output file in write mode
with open(output_file_file_name_nms_sf, "w") as file:
                # Iterate over the file data and write each entry to the file
    for file_name, number in file_name_nms_sf.items():
            file.write(f"{file_name}: {number}\n")

# print(f"Data saved to {output_file}")
            # plt.imshow(image)
            # plt.imshow(all_masks, alpha=0.5)
            # plt.show()