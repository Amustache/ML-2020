eval "$(conda shell.bash hook)"
conda activate test2
if [ -d "output/train" ]
then
    echo "Found train set"
else
    echo "Getting training data"
    python youtube_parser.py -url "https://www.youtube.com/watch?v=g2poG9zjEkc" -i "train_snes" -o "output/train"
    python youtube_parser.py -url "https://www.youtube.com/watch?v=L3kdaRe6M6U" -i "train_sega" -o "output/train"
    echo "Crop training data"
    python img_normalizer.py -ifolder "output/train" -nr True
    python youtube_parser.py -url "https://www.youtube.com/watch?v=i7pLUpidR8k&ab_channel=JVCom" -i "train_sega_t2" -o "output/train"
	python youtube_parser.py -url "https://www.youtube.com/watch?v=K6w0NXzeLVc&ab_channel=wizzgamer" -i "train_snes_t2" -o "output/train"
    echo "Resize training data"
    python img_normalizer.py -ifolder "output/train" -x 112 -y 160 -nc True
    echo "Done"
fi
if [ -d "output/test" ]
then
    echo "Found test set"
else
    echo "Getting testing data"
    python youtube_parser.py -url "https://www.youtube.com/watch?v=f_M5ZTSSdVc" -i "test_snes" -o "output/test"
    echo "Resizing testing data"
    python img_normalizer.py -ifolder "output/test" -x 112 -y 160
    echo "Done"
fi
