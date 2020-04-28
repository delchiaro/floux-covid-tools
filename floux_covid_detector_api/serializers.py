import base64
import io

from rest_framework import serializers
from PIL import Image

from floux_mask_detector.mask_processor import FrameMaskProcessor


class FaceBoundingBox(object):
    def __init__(self, top_left, bottom_right, label):
        self.top_left_x = top_left[0]
        self.top_left_y = top_left[1]
        self.bottom_right_x = bottom_right[0]
        self.bottom_right_y = bottom_right[1]
        self.label = label
        

class FaceBoundingBoxSerializer(serializers.Serializer):
    top_left_x = serializers.IntegerField()
    top_left_y = serializers.IntegerField()
    bottom_right_x = serializers.IntegerField()
    bottom_right_y = serializers.IntegerField()
    label = serializers.CharField()

    def create(self, validated_data):
        return FaceBoundingBox(**validated_data)

    def update(self, instance, validated_data):
        pass


class FrameSerializer(serializers.Serializer):
    frame = serializers.CharField(required=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_processor = FrameMaskProcessor('resnet50-mafa', 1, 'floux_mask_detector/ckpt/')
        self.mask_processor.load_model(gpu=0)

    def create(self, validated_data):
        img_data = base64.b64decode(validated_data.get('frame'))
        if not img_data:
            raise serializers.ValidationError('The frame param is not a base64 string')

        import cv2
        import numpy as np
        np_image_vector = np.frombuffer(img_data, dtype=np.uint8);
        np_image = cv2.imdecode(np_image_vector, flags=1)

        bboxes, labels, processed_img = self.mask_processor.predict_bboxes(np_image)
        # image_binary = Image.open(io.BytesIO(img_data))

        if len(bboxes) > 0:
            bounding_boxes = [FaceBoundingBox(bbox[:2], bbox[2:4], l) for bbox, l in zip(bboxes, labels)]
        else:
            bounding_boxes = [FaceBoundingBox(top_left=[0, 0], bottom_right=[0, 0], label='None')]
        return FaceBoundingBoxSerializer(bounding_boxes[0]).data

    def update(self, instance, validated_data):
        pass

