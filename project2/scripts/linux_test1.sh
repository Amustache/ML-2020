eval "$(conda shell.bash hook)"
conda activate ml_proj2
if [ -d "output/train" ]
then
    echo "Found train set"
else
    echo "Getting training data"
    python youtube_parser.py -url "https://www.youtube.com/watch?v=g2poG9zjEkc" -i "train_snes" -o "output/train"
    python youtube_parser.py -url "https://www.youtube.com/watch?v=L3kdaRe6M6U" -i "train_sega" -o "output/train"
    echo "Resizing training data"
    python img_normalizer.py -ifolder "output/train" -x 224 -y 320
    echo "Done"
fi
if [ -d "output/test" ]
then
    echo "Found test set"
else
    echo "Getting testing data"
    python youtube_parser.py -url "https://www.youtube.com/watch?v=f_M5ZTSSdVc" -i "test_snes" -o "output/test"
    echo "Resizing testing data"
    python img_normalizer.py -ifolder "output/test" -x 224 -y 320
    echo "Done"
fi
