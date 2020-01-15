mkdir -p output/syn
mkdir -p output/real
# synthetic
for i in $(seq 0 9);
do
    echo $i;
    python main.py \
        --input-left ./custom_data/Synthetic/TL$i.png \
        --input-right ./custom_data/Synthetic/TR$i.png \
        --output ./output/syn/TL$i.pfm
    python eval.py \
        --pred ./output/syn/TL$i.pfm \
        --gt ./custom_data/Synthetic/TLD$i.pfm
done
# real
for i in $(seq 0 9);
do
    echo $i;
    python main.py \
        --input-left ./custom_data/Real/TL$i.bmp \
        --input-right ./custom_data/Real/TR$i.bmp \
        --output ./output/real/TL$i.pfm
done