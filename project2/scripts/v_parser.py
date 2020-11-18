import argparse
import cv2
import os
import sys

def checkIfFoldersExists(output):
    cwd = os.getcwd()
    folders = output.split("/")
    for f in folders:
        # Last element is an empty string
        if f=="":
            break
        cwd = os.path.join(cwd, f)
        # If directory doesn't exist, create it
        if (os.path.isdir(cwd) == False):
            os.mkdir(cwd, 0o777)

def extractImages(filename, img_name, img_format, rate, output):
    checkIfFoldersExists(output)
    tmp_rate = rate
    vid = cv2.VideoCapture(filename)
    i=0
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == False:
            break
        if tmp_rate == 0:
            cv2.imwrite(output+img_name+str(i)+'.'+img_format, frame)
            tmp_rate = rate
            i = i+1
        else:
            tmp_rate = tmp_rate-1
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract images from videos')
    parser.add_argument('-fname', type=str, required=True, help='Name of video file to extract')
    parser.add_argument('-i', '--img', type=str, required=False, default='snapshot', help='Names of images saved. (default: snapshot)')
    parser.add_argument('-r', '--rate', type=int, required=False, default=10, help='Image rate to extract. If rate is 3, will save one every 3 images. (default: 10)')
    parser.add_argument('-f', '--iformat', type=str, required=False, default='png', help='Format of generated images. (default: png)')
    parser.add_argument('-o', '--output', type=str, required=False, default='output/', help='Output folder for images. (default: output/)')

    args = parser.parse_args(sys.argv[1:])
    extractImages(args.fname, args.iformat, args.rate, args.output)
