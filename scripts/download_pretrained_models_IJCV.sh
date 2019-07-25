mkdir -p data/
model="TOM-Net_plus_Bg_Model.tgz"

# Download pre-trained model
for model in "TOM-Net_plus_Bg_Model.tgz" "TOM-Net_plus_Trimap_Model.tgz"; do
    cd data/
    wget http://www.visionlab.cs.hku.hk/data/TOM-Net/IJCV_Extension/${model}
    tar -zxvf ${model}
    rm ${model}
done

# Back to root directory
cd ../
