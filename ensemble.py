import pandas as pd
from scipy.stats import rankdata



pred = 0.8 * ranked("../input/deletion_specific_ensemble/submission.csv") + \
 0.2 * ranked("submission.csv")


df = pd.read_csv('../input/novozymes-enzyme-stability-prediction/sample_submission.csv')
df.tm = pred


# equally weighted ensemble with https://www.kaggle.com/code/shlomoron/nesp-relaxed-rosetta-scores
#df.tm = rankdata(df.tm) + ranked('../input/nesp-relaxed-rosetta-scores/submission_rosetta_scores')


df.to_csv('ensemble_submission_w_603.csv', index=False)
