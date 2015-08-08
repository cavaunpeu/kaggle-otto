import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from utils import BoostedTreesClassifier
from params import params_list


def shuffle_df(df):
    rnd = np.random.RandomState(12345)
    df_shuff = df.reindex(rnd.permutation(df.index))
    return df_shuff.reset_index(drop=True)


if __name__ == '__main__':

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    sample_submission_path = sys.argv[3]
    submission_path = sys.argv[4]

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_submission_path)

    train = shuffle_df(train)
    X_test = test.drop('id', axis=1)
    X, y = train.drop(['id', 'target'], axis=1), train['target']

    preds = []
    for i, params in enumerate(params_list):
        print('Now fitting, predicting with params_{} of {}'.format(str(i+1), str(len(params_list))))
        clf_1 = BoostedTreesClassifier(**params)
        clf_1.fit(X, y)
        preds_1 = clf_1.predict_proba(X_test)

        clf_2 = CalibratedClassifierCV(BoostedTreesClassifier(**params), method='isotonic', cv=10)
        clf_2.fit(X, y)
        preds_2 = clf_2.predict_proba(X_test)

        preds.extend([preds_1, preds_2])

    leaderboard_preds = np.mean(np.array(preds), axis=0)
    sub = pd.DataFrame(leaderboard_preds, index=sample['id'].values, columns=sample.columns[1:])
    sub.to_csv(submission_path, index_label='id')
