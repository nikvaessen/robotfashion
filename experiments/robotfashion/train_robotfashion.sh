python3 ../../robotfashion/robotfashion/models/faster_rcnn/trainer.py \
  --gpus 0 \
  --nodes 1 \
  --num-data-loaders 4 \
  --max_nb_epochs 100 \
  --batch-size 8 \
  --data-folder-path .. \
  --save-weights-every-n 1 \
  --freeze-backbone True \
  --dataset "robotfashion"

python3 ../../robotfashion/robotfashion/models/faster_rcnn/trainer.py \
  --gpus 0 \
  --nodes 1 \
  --num-data-loaders 4 \
  --max_nb_epochs 100 \
  --batch-size 8 \
  --data-folder-path .. \
  --save-weights-every-n 1 \
  --freeze-backbone False \
  --dataset "robotfashion"

sudo shutdown now