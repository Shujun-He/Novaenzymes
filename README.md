# Novaenzymes


# Download data

```bash setups.h``` to download all necessary data \
```python make_features.py``` to preprocess pdbs files into features

# Environment

Use ```environment.yml``` to create the same environment I use


# To run: ```train.sh```

I ran the training on a server with 8xA6000 so I ran 8 folds concurrently. You will want to change that based on how many GPUs you have.

To run one fold: ```bash run_one_fold.sh```



