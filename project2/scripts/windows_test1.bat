IF EXIST output\train\NUL (
    echo Found training data
) ELSE (
    echo Getting trainig data
    python youtube_parser.py -url "https://www.youtube.com/watch?v=g2poG9zjEkc" -i "train_snes" -o "output/train"
    python youtube_parser.py -url "https://www.youtube.com/watch?v=L3kdaRe6M6U" -i "train_sega" -o "output/train"
    echo Resizing training data
    python img_normalizer.py -ifolder "output/train" -x 224 -y 320
    echo Done
)
IF EXIST output\test\NUL (
    echo Found testing data
) ELSE (
    echo Getting testig data
    python youtube_parser.py -url "https://www.youtube.com/watch?v=f_M5ZTSSdVc" -i "test_snes" -o "output/test"
    python youtube_parser.py -url "https://www.youtube.com/watch?v=WCzPGddOWqc" -r 30 -i "test_gba" -o "output/test"
    echo Resizing testing data
    python img_normalizer.py -ifolder "output/test" -x 224 -y 320
    python youtube_parser.py -url "https://www.youtube.com/watch?v=U5f-ri5dtEU&ab_channel=wizzgamer" -i "test_sega" -o "output/test"
    echo Resizing testing data
    python img_normalizer.py -ifolder "output/test" -nc True -x 224 -y 320
    echo Done
)