import io
import queue
from copy import deepcopy
from os.path import join
from pprint import pprint

import PIL
from  matplotlib import patches
import cv2
import skimage
import torch
from pylab import *

from floux_mask_detector import csv_eval

from .dataloader import  normalize, resize




class FrameMaskProcessor:
    def __init__(self, model_name, model_epoch, checkpoint_folder):
        self.model_name = model_name
        self.model_epoch = model_epoch
        self.retinanet = None
        self.device = None
        self.capture = None
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])
        self.checkpoint_folder = checkpoint_folder


    def load_model(self, gpu=0):
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        import sys
        from floux_mask_detector import utils, model_level_attention, anchors, losses
        sys.modules['.utils'] = utils  # creates a packageA entry in sys.modules
        sys.modules['utils'] = utils  # creates a packageA entry in sys.modules
        sys.modules['anchors'] = anchors  # creates a packageA entry in sys.modules
        sys.modules['losses'] = losses  # creates a packageA entry in sys.modules
        sys.modules['.model_level_attention'] = model_level_attention  # creates a packageA entry in sys.modules


        retinanet = torch.load(join(self.checkpoint_folder, f'{self.model_name}_{self.model_epoch}.pt'),
                               map_location=self.device, )
        # print(retinanet)
        retinanet = retinanet.to(self.device)
        retinanet.training = False
        retinanet.eval()
        retinanet.freeze_bn()
        self.retinanet = retinanet

    def preprocess_image(self, image, transform=None):
        b, g, r = cv2.split(image)
        img = cv2.merge([r, g, b])
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        img = img.astype(np.float32) / 255.0
        if transform is not None:
            img = transform(img)
        return img

    def predict_bboxes(self, frame, verbose=False):
        orig_img = self.preprocess_image(frame)
        img, scale = resize(orig_img)
        img = normalize(img, self.mean, self.std)
        img = torch.Tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        pred = csv_eval.predict(self.retinanet, img, scale, nb_classes=3, max_detections=100, score_threshold=0.5)
        detections, (image_boxes, image_scores, image_labels, image_detections) = pred

        if verbose:
            pprint(detections)
            pprint(image_boxes)
            pprint(image_scores)
            pprint(image_labels)
            pprint(image_detections)
            print('\n\n\END..\n')

        labels_dict = {0: 'simple_mask',
                       1: 'complex_mask',
                       2: 'human_body'}

        return image_boxes, [labels_dict[l] for l in image_labels], img.detach().cpu().numpy()


    def process_frame(self, frame, plot=False, output_queue: queue.Queue=None, verbose=False):
        orig_img = self.preprocess_image(frame)
        img, scale = resize(orig_img)
        img = normalize(img, self.mean, self.std)
        img = torch.Tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        pred = csv_eval.predict(self.retinanet, img, scale, nb_classes=3, max_detections=100, score_threshold=0.5)
        detections, (image_boxes, image_scores, image_labels, image_detections) = pred

        if verbose:
            pprint(detections)
            pprint(image_boxes)
            pprint(image_scores)
            pprint(image_labels)
            pprint(image_detections)
            print('\n\n\END..\n')

        labels_dict = {0: 'simple_mask',
                       1: 'complex_mask',
                       2: 'human_body'}

        fig, ax = plt.subplots(1)
        ax.imshow(orig_img)
        for bbox, lbl in zip(image_boxes, image_labels):
            x, y = bbox[0:2]
            w = bbox[2] - x
            h = bbox[3] - y
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            label = text(x, y, labels_dict[lbl], fontsize=22, color='red')
            ax.add_patch(rect)

        if plot:
            fig.show()

        if output_queue is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            pil_img = deepcopy(PIL.Image.open(buf))
            buf.close()
            try:
                output_queue.get_nowait()
            except queue.Empty:
                pass
            output_queue.put_nowait(pil_img)

        return fig
#
# # GPU = 0
# # # BS = 1
# # MODEL_NAME = 'resnet50-mafa'
# # MODEL_EPOCH = 1
# mask_processor = StreamingMaskProcessor(parser.model_name, parser.model_epoch, parser.url)
# mask_processor.input_streaming_url()
# mask_processor.load_model(parser.gpu)
# mask_processor.start_processing_loop(verbose=True)