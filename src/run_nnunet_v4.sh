export prev_dice=0
source ~/miniconda3/bin/activate
rm ./output.log
python3 nnunet_v4.py > ./output.log 2>&1