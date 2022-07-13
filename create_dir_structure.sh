mv training train
mv train/groundtruth train/gt

mkdir train_crops
mkdir train_crops/gt
mkdir train_crops/images

mkdir val
mkdir val/images
mkdir val/gt
mkdir val_crops
mkdir val_crops/images
mkdir val_crops/gt

rm train.txt
for i in {1..101}
do
    echo $i >> train.txt
done

rm val.txt
for i in {102..144}
do
    echo $i >> val.txt
    cp train/images/satimage_$i.png val/images/ 
    cp train/gt/satimage_$i.png val/gt/ 
done
