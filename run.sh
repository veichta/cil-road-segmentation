for w in 0 0.1 0.05 0.01 0.005 0.001
do 
    for b in 1 0
    do
        python main.py \
            --model spin \
            --device cuda \
            --data_path /mnt/ds3lab-scratch/veichta/road-seg/big-dataset \
            --num_epochs 200 \
            --batch_size 4 \
            --lr 0.01 \
            --weight_miou 1 \
            --weight_vec 0.5 \
            --weight_topo $w \
            --weight_bce 0 \
            --weight_dice 0 \
            --val_split 0.2 \
            --augmentations $b
    done
done