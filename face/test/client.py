#!/usr/bin/env python

import argparse
import numpy as np
import sys
import cv2

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

#from processing import preprocess, postprocess

from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import COCOLabels

INPUT_NAMES = ["images"]
OUTPUT_NAMES = ["output"]

def extract_boxes(predictions, scale):
    # self.scale = np.array(
    #         [self.img_width / self.input_width, self.img_height / self.input_height, self.img_width / self.input_width,
    #          self.img_height / self.input_height], dtype=np.float32)
    
    #scale = np.array([1,1,1,1], dtype=np.float32)
    
    # Extract boxes from predictions
    boxes = predictions[:, :4] * scale
    kpts = predictions[:, 6:]  ###x1,y1,score1, ...., x5,y5,score5
    kpts *= np.tile(np.array([scale[0], scale[1], 1], dtype=np.float32), (1, 5))

    # Convert boxes to xywh format
    boxes_ = np.copy(boxes)
    boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
    boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
    return boxes_, kpts

def process_output(predictions, scale):
    conf_threshold=0.2
    iou_threshold=0.5
    
    # Filter out object confidence scores below threshold
    obj_conf = predictions[:, 4]
    predictions = predictions[obj_conf > conf_threshold]
    obj_conf = obj_conf[obj_conf > conf_threshold]

    # Multiply class confidence with bounding box confidence
    predictions[:, 5] *= obj_conf

    # Get the scores
    scores = predictions[:, 5]

    # Filter out the objects with a low score
    valid_scores = scores > conf_threshold
    predictions = predictions[valid_scores]
    scores = scores[valid_scores]

    # Get bounding boxes for each object
    boxes, kpts = extract_boxes(predictions, scale)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    return boxes[indices], scores[indices], kpts[indices]

def draw_detections(image, boxes, scores, kpts):
    for box, score, kp in zip(boxes, scores, kpts):
        
        # x, y, w, h = box.astype(int)
        x, y, w, h = box.astype(int)[0]
        kp = kp[0]
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        label = "face"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        for i in range(5):
            cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 1, (0, 255, 0), thickness=-1)
    return image

def prepare_input(image):
    input_width, input_height = 640, 640
    img_height, img_width = image.shape[:2]
    scale = np.array(
        [img_width / input_width, img_height / input_height, img_width / input_width,
            img_height / input_height], dtype=np.float32)
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize input image
    input_img = cv2.resize(input_img, (input_width, input_height))

    # Scale input pixel values to 0 to 1
    input_img = input_img.astype(np.float32) / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :]
    return img_height, img_width, scale, input_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['dummy', 'image', 'video'],default='dummy',
                        help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
    parser.add_argument('input',type=str,nargs='?',
                        help='Input file to load from in image or video mode')
    parser.add_argument('-m','--model',type=str,required=False,default='yolov7',
                        help='Inference model name, default yolov7')
    parser.add_argument('--width',type=int,required=False,default=640,
                        help='Inference model input width, default 640')
    parser.add_argument('--height',type=int,required=False,default=640,
                        help='Inference model input height, default 640')
    parser.add_argument('-u','--url',type=str,required=False,default='localhost:8001',
                        help='Inference server URL, default localhost:8001')
    parser.add_argument('-o','--out',type=str,required=False,default='',
                        help='Write output into file instead of displaying it')
    parser.add_argument('-f','--fps',type=float,required=False,default=24.0,
                        help='Video output fps, default 24.0 FPS')
    parser.add_argument('-i','--model-info',action="store_true",required=False,default=False,
                        help='Print model status, configuration and statistics')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose client output')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds, default no timeout')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates, default none')
    parser.add_argument('-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key, default is none')
    parser.add_argument('-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none')

    FLAGS = parser.parse_args()

    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(FLAGS.model):
        print("FAILED : is_model_ready")
        sys.exit(1)

    if FLAGS.model_info:
        # Model metadata
        try:
            metadata = triton_client.get_model_metadata(FLAGS.model)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = triton_client.get_model_config(FLAGS.model)
            if not (config.config.name == FLAGS.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)

    print("Current mode is", FLAGS.mode)
    # Draw detections
    import time
    start_time = time.time()

    # IMAGE MODE
    if FLAGS.mode == 'image':
        print("Running in 'image' mode")
        if not FLAGS.input:
            print("FAILED: no input image")
            sys.exit(1)

        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, FLAGS.width, FLAGS.height], "FP32"))        
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))

        print("Creating buffer from image file...")
        input_image = cv2.imread(str(FLAGS.input))
        if input_image is None:
            print(f"FAILED: could not load input image {str(FLAGS.input)}")
            sys.exit(1)
                
        # input_image_buffer = preprocess(input_image, [FLAGS.width, FLAGS.height])
        # input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        # inputs[0].set_data_from_numpy(input_image_buffer)
        
        img_height, img_width, scale, input_image_buffer = prepare_input(input_image)
        inputs[0].set_data_from_numpy(input_image_buffer)

        print("Invoking inference...")
        results = triton_client.infer(model_name=FLAGS.model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=FLAGS.client_timeout)
        
    
        if FLAGS.model_info:
            statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
                sys.exit(1)
            print(statistics)
        print("Done")
        
        output = results.as_numpy(OUTPUT_NAMES[0])[0]
        print(f"Received result buffer \"{OUTPUT_NAMES[0]}\" of size {output.shape}")
        print(f"Naive buffer sum: {np.sum(output)}")
    
        boxes, scores, kpts = boxes, scores, kpts = process_output(output, scale)

        dstimg = draw_detections(input_image, boxes, scores, kpts)
        
        winName = 'Deep learning object detection in ONNXRuntime'

        print("--- %s seconds ---" % (time.time() - start_time))

        cv2.imwrite("./data/selfie_result.jpg", dstimg)
        