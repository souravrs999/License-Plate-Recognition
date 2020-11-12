'''
utf8 python3
Time       : 20/11/12
Author     : Sourav R S
File       : main.pyi

'''

''' Necessary imports '''
import cv2
import argparse
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

def run_inference(cfgfile, weightfile):
    model = Darknet(cfgfile)
    model.print(cfgfile)
    
    ''' Throws error if could not load weight file '''
    try:
        model.load(weightfile)
    except Exception:
        print("Could not load Weights")

    ''' Check if cuda is available '''
    cuda = torch.cuda.is_available()

    ''' If GPU is available load the model onto the GPU '''
    if cuda:
        model.cuda()

    ''' Grab the frame from source '''
    cap = cv2.VideoCapture(source):
        cap.set(3, 1280)
        cap.set(4, 720)

    ''' Load the labels from the .names file '''
    class_names = load_class_names(class_names)

    while True:
        ret, img = cap.read()

        ''' Checked to see frame received successfully '''
        if not ret:
            exit(0)

        ''' Resize the image to those specified in the configuration file '''
        img_resized = cv2.resize(img, (model.width, model.height))
        img  = cv2.cvtColor(img_resized, cv2.COLOR_BG2RGB)
        start = time.time()
        boxes = do_detect(model, img, 0.4, 0.6, cuda)
        end = time.time()
        print("predicted in %f seconds." % (end - start))

        ''' Returns the annotated images '''
        antd_img = plot_boxes(img, boxes[0], save_name=None, class_names=class_names)

        cv2.imshow('Inference', antd_img)
        cv2.waitkey(0.1)

    ''' Release the frame '''
    cap.release()
