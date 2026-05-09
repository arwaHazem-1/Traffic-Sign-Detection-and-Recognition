uhhh download the dataset 
https://www.kaggle.com/datasets/tuanai/traffic-signs-dataset 

 /\_/\
( o.o )
 > ^ <

to train the model
python evaluation.py --image_dir "data/train" --labels "data/labels.csv" --output_dir "eval_results"

to test on one image
python predict.py --image "data/test/10/010_1_0012_1_j.png" --model "eval_results/classifier.pkl"

Visualize Pipeline
python main.py --image "data/test/6/006_1_0003_1_j.png" --output_dir "main_results"
