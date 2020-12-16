import argparse
import pytube
import os
import sys

from v_parser import extractImages

parser = argparse.ArgumentParser(description='Extract images from a Youtube url')
parser.add_argument('-url', type=str, required=True, help='Youtube url of video to extract')
parser.add_argument('-i', '--img', type=str, required=False, default='snapshot', help='Names of images saved. (default: snapshot)')
parser.add_argument('-r', '--rate', type=int, required=False, default=10, help='Image rate to extract. If rate is 3, will save one every 3 images. (default: 10)')
parser.add_argument('-f', '--iformat', type=str, required=False, default='png', help='Format of generated images. (default: png)')
parser.add_argument('-o', '--output', type=str, required=False, default='output/', help='Output folder for images. (default: output/)')

args = parser.parse_args(sys.argv[1:])

url = args.url
img_name = args.img
img_format = args.iformat
rate = args.rate
output = args.output

youtube = pytube.YouTube(url)
video = youtube.streams.get_highest_resolution()
v_title = (video.title).replace(" ", "_").replace("/", "")
video.download(os.path.join(os.getcwd(), '..', 'data', 'input', 'videos'), filename=v_title)

extractImages('videos/'+v_title+'.'+video.subtype, img_name, img_format, rate, output)
