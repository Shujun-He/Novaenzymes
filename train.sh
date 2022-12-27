for i in {0..7};do
  nohup python train.py --fold $i --nfolds 8 --gpu_id $i --workers 8 > ${i}.out &
done
