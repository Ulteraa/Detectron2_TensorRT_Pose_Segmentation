import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import time
import tritonclient.http as httpclient
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import torch
from postprocessor import paste_masks_in_image

HEIGHT = 624
WIDTH = 1008
THRESHOLD = 0.5

def test_infer(req_id, image_file, model_name, print_output=False):
    
    img = cv2.imread(image_file)
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
    img = np.ascontiguousarray(img.astype("float32").transpose(2,0,1))
    #img = np.array(Image.open(image_file))
    #img = np.ascontiguousarray(img.transpose(2, 0, 1))
    
    # Define model's inputs
    inputs = []
    inputs.append(httpclient.InferInput('image__0', img.shape, "FP32"))
    inputs[0].set_data_from_numpy(img)
    # Define model's outputs
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('bboxes__0'))
    outputs.append(httpclient.InferRequestedOutput('classes__1'))
    outputs.append(httpclient.InferRequestedOutput('masks__2'))
    outputs.append(httpclient.InferRequestedOutput('scores__3'))
    outputs.append(httpclient.InferRequestedOutput('shape__4'))
    # Send request to Triton server

    triton_client = httpclient.InferenceServerClient(
        url="localhost:8000", verbose=False)

    results = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    response_info = results.get_response()
    outputs = {}
    for output_info in response_info['outputs']:
        output_name = output_info['name']
        outputs[output_name] = results.as_numpy(output_name)

    return outputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default = 'tensorrt_model')
    parser.add_argument('--num-reqs', default='1')
    parser.add_argument('--print-output', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_file = args.image
    model_name = args.model
    n_reqs = int(args.num_reqs)

    outputs = test_infer(0, image_file, model_name, False)
    masks_list = outputs['masks__2']
    bboxes = outputs['bboxes__0']
    mask_img = paste_masks_in_image(masks_list, bboxes, (HEIGHT, WIDTH), THRESHOLD)
  

