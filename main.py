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

def str2int(source):

    ''' Converts the source obtained from arguments
        to int handles both videosources and webcam
        indexed '''
    try:
        return int(source)

    except ValueError:
        return source

def run_inference(cfgfile, weightfile, namesfile, source, output):
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

    ''' Switching model to eval mode and
        setting torch.no_grad '''
    model.eval()
    with torch.no_grad():

        ''' Grab the frame from source '''
        source = str2int(source)
        cap = cv2.VideoCapture(source)

        '''Get height width and frame rate of input video '''
        width = int(cap.get(3))
        height = int(cap.get(4))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        ''' Load the labels from the .names file '''
        class_names = load_class_names(namesfile)

        ''' Video writer '''
        if output:
            out = cv2.VideoWriter('output.mp4',
                    cv2.VideoWriter_fourcc('X','2','6','4'),
                    frame_rate, (width, height))

        while True:
            ret, img = cap.read()

            ''' Checked to see frame received successfully '''
            if not ret:
                exit(0)

            try:

                ''' Resize the image to those specified
                in the configuration file '''
                img_resized = cv2.resize(img, (model.width, model.height))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

                start = time.time()
                boxes = do_detect(model, img_rgb, 0.3, 0.6, cuda)

                print("predicted in %f seconds." % (time.time() - start))

                ''' Returns the annotated images '''
                antd_img = plot_boxes_cv2(img, boxes[0],
                        savename=None, class_names=class_names)

                '''Calculate the framerate for the inference loop '''
                fps = int(1/(time.time()-start))
                print(f"FPS: {fps}")

            except Exception:
                pass

            ''' Write frame into output.mp4 file '''
            if output:
                out.write(antd_img)

            ''' Implicitly create a named window '''
            cv2.namedWindow('Inference', cv2.WINDOW_NORMAL)

            ''' Show the frame '''
            cv2.imshow('Inference', antd_img)

            key = cv2.waitKey(1)

            ''' If the 'q' key is pressed break
                out of the loop '''
            if key & 0xFF == ord('q'):
                break

        ''' Release the frame and writer'''
        cap.release()
        if output:
            out.release()
        cv2.destroyAllWindows()

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
            type=str,
            default=0,
            help='Source for webcam default 0 for the built in webcam',
            dest='source')

    parser.add_argument('-output',
            type=bool,
            default=False,
            help='True/Flase if you want to output the result video',
            dest='output')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = arguments()
    run_inference(args.cfgfile, args.weightfile,
            args.namesfile, args.source,
            args.output)
