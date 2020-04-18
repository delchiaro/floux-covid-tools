import base64
import io

from rest_framework import serializers
from PIL import Image


class FaceBoundingBox(object):
    def __init__(self, top_left, bottom_right, contains_mask):
        self.top_left_x = top_left[0]
        self.top_left_y = top_left[1]
        self.bottom_right_x = bottom_right[0]
        self.bottom_right_y = bottom_right[1]
        self.contains_mask = contains_mask
        

class FaceBoundingBoxSerializer(serializers.Serializer):
    top_left_x = serializers.IntegerField()
    top_left_y = serializers.IntegerField()
    bottom_right_x = serializers.IntegerField()
    bottom_right_y = serializers.IntegerField()
    contains_mask = serializers.BooleanField()

    def create(self, validated_data):
        return FaceBoundingBox(**validated_data)

    def update(self, instance, validated_data):
        pass


class FrameSerializer(serializers.Serializer):
    frame = serializers.CharField(required=True)

    def create(self, validated_data):
        img_data = base64.b64decode(validated_data.get('frame'))
        if not img_data:
            raise serializers.ValidationError('The frame param is not a base64 string')
        # TODO: Process the image binary and return the bounding box
        # image_binary = Image.open(io.BytesIO(img_data))

        bounding_box = FaceBoundingBox(top_left=[0, 0], bottom_right=[0, 0], contains_mask=False)
        return FaceBoundingBoxSerializer(bounding_box).data

    def update(self, instance, validated_data):
        pass

