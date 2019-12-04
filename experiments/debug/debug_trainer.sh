# DEBUG_RUN 1=True, 0=False

python3 ../../robotfashion/robotfashion/models/faster_rcnn/trainer.py \
  --gpus 0 \
  --nodes 1 \
  --max_nb_epochs 20 \
  --debug 1 \
  --batch-size 1 \
  --data-folder-path .. \
  --overfit_pct 0.001

python3 ../../robotfashion/robotfashion/models/faster_rcnn/trainer.py \
  --gpus 0 \
  --nodes 1 \
  --max_nb_epochs 20 \
  --debug 0 \
  --batch-size 1 \
  --data-folder-path .. \
  --overfit_pct 0.0001