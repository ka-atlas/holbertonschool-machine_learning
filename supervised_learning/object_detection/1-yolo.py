#!/usr/bin/env python3
"""Yolo Task 1"""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """
    Yolo class uses algorithm Yolo v3 to complete
    object detection in images and videos.
    Objects are classified within a frame.
    The purpose of this class is to allow for user-friendly
    usage of YOLOv3 by encapsulating model loading,
    class info, and parameter config.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Adding expected parameters
           Args:
                model_path: path to pretained Yolo model
                classes_path: list of class names
                class_t: box score threshold for filtering
                nms_t: IOU threshold for non max suppression
                anchors: anchor for box info
        """
        self.model = K.models.load_model(model_path)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        with open(classes_path) as file:
            class_names = file.read()
        self.class_names = class_names.replace("\n", "|").split("|")[:-1]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Processes Outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            
            # Box coordinates adjustment (output[..., 0:1] is the same as output[..., 0], and output[..., 0] is simpler to read)
            box_tx = output[..., 0]
            box_ty = output[..., 1]
            box_tw = output[..., 2]
            box_th = output[..., 3]
            
            # Using sigmoid function
            box_tx_sigmoid = self.sigmoid(box_tx)
            box_ty_sigmoid = self.sigmoid(box_ty)
            
            # Create a grid of same shape as the predictions (the previous code was very hard to read and I wasn't sure whether it produced the correct result - I rewrote it)
            grid_x = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_y = np.arange(grid_height).reshape(1, grid_height, 1)

            # x,y coordinates of the boxes need to be divided by the width and height of the grid
            box_x = (box_tx_sigmoid + grid_x)/grid_width
            box_y = (box_ty_sigmoid + grid_y)/grid_height

            # width and height of the boxes need to be divided by the input width and height (not the grid's height and width!)
            # self.anchors[:, 0] ti self.anchors[i, :, 0] as you need to multiply the output with it's respective bounding box and not the whole array
            box_w = (self.anchors[i, :, 0] * np.exp(box_tw))/input_width
            box_h = (self.anchors[i, :, 1] * np.exp(box_th))/input_height
            
            # Convert coordinates relative to the size of the image
            x1 = (box_x - box_w / 2) * image_size[1]
            y1 = (box_y - box_h / 2) * image_size[0]
            x2 = (box_x + box_w / 2) * image_size[1]
            y2 = (box_y + box_h / 2) * image_size[0]

            # # Box confidences and class probabilities
            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_conf = self.sigmoid(output[..., 4:5])
            box_confidences.append(box_conf)
            box_class_prob = self.sigmoid(output[..., 5:])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
