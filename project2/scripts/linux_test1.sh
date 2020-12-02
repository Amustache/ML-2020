eval "$(conda shell.bash hook)"
conda activate ml_proj2
if [ -d "output/train" ]
then
    echo "found"
else
    echo "get data"
fi
