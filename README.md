uhhh download the dataset 
https://www.kaggle.com/datasets/tuanai/traffic-signs-dataset 

 /\_/\
( o.o )
 > ^ <


python evaluation.py --image_dir "C:\Users\youss\Downloads\dataset\archivedataset\DATA" --labels "C:\Users\youss\Downloads\dataset\archivedataset\labels.csv" --output_dir eval_result

python predict.py --image "C:\Users\youss\Downloads\dataset\archivedataset\DATA\14\00001.ppm" --model "eval_result\classifier.pkl"
