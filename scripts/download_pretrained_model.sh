mkdir -p data/
model="TOM-Net_model.tgz"

# Download pre-trained model
cd data/
wget http://www.visionlab.cs.hku.hk/data/TOM-Net/${model}
tar -zxvf ${model}
rm ${model}

# Back to root directory
cd ../
