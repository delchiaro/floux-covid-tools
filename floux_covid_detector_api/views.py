from rest_framework import views, status
from rest_framework.response import Response

from floux_covid_detector_api.serializers import FrameSerializer


class ProcessFrameView(views.APIView):
    def post(self):
        frame_serializer = FrameSerializer(data=self.request.data)
        frame_serializer.is_valid(raise_exception=True)
        response_data = frame_serializer.save()
        return Response(response_data, status=status.HTTP_200_OK)


