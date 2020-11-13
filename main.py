'''
utf-8 python3
Time       : 20/11/12
Author     : Sourav R S
File       : main.py

'''

''' Necessary imports '''
import cv2
import argparse
from tools.utils import *
from tools.torch_utils import *
from tools.darknet2pytorch import Darknet

def run_inference(cfgfile, weightfile, namesfile, source):
    model = Darknet(cfgfile)
    model.print_network()

    ''' Throws error if could not load weight file '''
    try:
        model.load_weights(weightfile)
    except Exception:
        print("Could not load Weights")

    ''' Check if cuda is available '''
    cuda = torch.cuda.is_available()

    ''' If GPU is available load the model onto the GPU '''
    if cuda:
        model.cuda()

    ''' Grab the frame from source
        If you want to test it on a video uncomment
        the following add the path to the file '''

    source = "./videoplayback.mp4"
    cap = cv2.VideoCapture(source)
    cap.set(3, 1280)
    cap.set(4, 720)

    ''' Load the labels from the .names file '''
    class_names = load_class_names(namesfile)

    frames = 0
    while True:
        ret, img = cap.read()

        ''' Checked to see frame received successfully '''
        if not ret:
            exit(0)

        ''' Resize the image to those specified
        in the configuration file '''
        img_resized = cv2.resize(img, (model.width, model.height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(model, img_rgb, 0.4, 0.6, cuda)

        print("predicted in %f seconds." % (time.time() - start))

        ''' Returns the annotated images '''
        antd_img = plot_boxes_cv2(img, boxes[0],
                savename=None, class_names=class_names)

        frames += 1

        '''Calculate the framerate for the inference loop '''
        print(f"FPS: {int(frames/(time.time()-start))}")
        cv2.imshow('Inference', antd_img)
        key = cv2.waitKey(1)

        ''' If the 'q' key is pressed break
            out of the loop '''
        if key & 0xFF == ord('q'):
            break

    ''' Release the frame '''
    cap.release()

def arguments():

    ''' Arguments for running infernce '''
    parser = argparse.ArgumentParser("Arguments for running inference.")

    parser.add_argument('-cfgfile',
            type=str,
            default='./cfg/yolov4.cfg',
            help='Path to the configuration file',
            dest='cfgfile')

    parser.add_argument('-weightfile',
            type=str,
            default='./weights/yolov4.weights',
            help='Path to the weights file',
            dest='weightfile')

    parser.add_argument('-namesfile',
            type=str,
            default='./cfg/classes.names',
            help='Path to the classes name file',
            dest='namesfile')

    parser.add_argument('-source',
            type=int,
            default=0,
            help='Source for webcam default 0 for the built in webcam',
            dest='source')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = arguments()
    run_inference(args.cfgfile, args.weightfile,
            args.namesfile, args.source)
