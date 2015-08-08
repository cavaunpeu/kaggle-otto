## [Otto Group Production Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)

#### Summary
* I experimented with many models, and Gradient Boosted Trees performed best
* Feature engineering in this competition was ~fruitless
* Ample effort was spent tuning hyperparameters of my models. The parameters chosen were a result of cross-validated grid search over a wide parameter space
* Ensembling wins Kaggle competitions. Here, I average the predictions of 88 different models

#### To train models and generate a submission, run the following on the command line:

```bash
python create_submission.py <path_to_training_set> <path_to_test_set> <path_to_sample_submission> <path_to_your_submission>
```

#### This should score roughly .436 on the leaderboard, which lands in the top ~10%.