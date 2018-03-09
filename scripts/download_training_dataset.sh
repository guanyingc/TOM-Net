mkdir -p data/datasets
cd data/datasets

# Download real testing dataset
dataset="TOM-Net_Synth_Train_178k.tgz"
wget http://www.visionlab.cs.hku.hk/data/TOM-Net/$dataset
tar -zxvf $dataset
rm $dataset

# Back to root directory
cd ../../

