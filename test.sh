for i in {0..0};do
  python train.py --fold $i --nfolds 8 --gpu_id $i --workers 8 --destabilizing_mutations_only
done
