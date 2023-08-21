

models="inception_v3 nasnet randwire"
opt_types="hios_lp hios_mr"
cnt=1
ngpu=2
bs=1
for model in $models; do
        for dm in  331 512 768 1280; do
                for opt_type in $opt_types; do
            for index in $(seq $cnt); do
                python main.py --device v100 --model $model --bs $bs --opt_type $opt_type --index $index --height $dm --width $dm --ngpu $ngpu
            done
                done
        done
done










