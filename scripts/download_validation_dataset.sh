mkdir -p data/datasets
cd data/datasets

# Download synthetic evaluation dataset
dataset="TOM-Net_Synth_Val_900.tgz"
wget http://www.visionlab.cs.hku.hk/data/TOM-Net/$dataset
tar -zxvf $dataset
rm $dataset

# Back to root directory
cd ../../

