
## Learning controller from STL specifications (traffic case)


### Training
python train_traffic.py -e traffic_demo --lr 5e-4

### Testing
python train_traffic.py -e traffic_demo --lr 5e-4 --test -P xxx/models/model_49000.ckpt