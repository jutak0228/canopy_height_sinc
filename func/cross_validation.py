import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score, precision_score


class CrossValidation():

    def __init__(self, X, y, n_cv, resample_range=[1, 2]):
        self.X = X
        self.y = y
        self.n_cv = n_cv
        self.resample_lower = resample_range[0]
        self.resample_upper = resample_range[1]
        self.n_resample = self.resample_upper - self.resample_lower + 1
        self.metrics = {
            'precision': precision_score,
            'f1_score': f1_score,
            'kappa_score': cohen_kappa_score}

    def search_thr(self, model, X_val, y_val, evaluation, p_upper):
        candidates = np.arange(0, p_upper, 0.025)
        metrics_val = np.empty_like(candidates)
        score = self.metrics[evaluation]
        for i_thr, thr in enumerate(candidates):
            y_val_pred = model.predict_proba(X_val)[:, 1] > thr
            metrics_val[i_thr] = score(y_val, y_val_pred)

        thr_best = candidates[np.argmax(metrics_val)]

        return thr_best

    def calculate(self, evaluation='precision', p_upper=0.8):
        clfs = {'LR': LogisticRegression()}
        n_clf = len(clfs)
        col_res = [
            'ratio',
            'i_cv',
            'clf',
            'thr',
            'support',
            'precision',
            'recall',
            'f1-score',
            'accuracy',
            'kappa']
        res = pd.DataFrame(
            index=range(
                self.n_cv *
                self.n_resample *
                n_clf),
            columns=col_res)

        kf = KFold(n_splits=self.n_cv, shuffle=True, random_state=25)
        i_row = 0
        for i_cv, (index_train_cv, test_index_cv) in enumerate(
                kf.split(self.y)):
            X_train, y_train = self.X[index_train_cv], self.y[index_train_cv]
            X_test_val, y_test_val = self.X[test_index_cv], self.y[test_index_cv]

            X_test, X_val, y_test, y_val = train_test_split(
                X_test_val, y_test_val, test_size=0.5)
            n_soft_train = len(y_train) - y_train.sum()

            for r_resample in range(
                    self.resample_lower,
                    self.resample_upper + 1):
                index_train = np.arange(len(index_train_cv))
                index_train_hard = index_train[y_train == 1]
                index_train_soft = index_train[y_train == 0]
                index_train_soft_rs = np.random.choice(
                    index_train_soft, replace=True, size=r_resample * n_soft_train)
                index_train_rs = np.concatenate(
                    [index_train_soft_rs, index_train_hard])

                X_train_rs, y_train_rs = X_train[index_train_rs], y_train[index_train_rs]

                for key_clf in clfs.keys():
                    clf = clfs[key_clf]
                    model = clf.fit(X_train_rs, y_train_rs)
                    thr_best = self.search_thr(
                        model, X_val, y_val, evaluation, p_upper)
                    y_test_pred = model.predict_proba(X_test)[:, 1] > thr_best
                    score_all = classification_report(
                        y_test, y_test_pred, output_dict=True, zero_division=1)
                    score = score_all['1'].copy()

                    score['i_cv'] = i_cv
                    score['ratio'] = r_resample
                    score['clf'] = key_clf
                    score['thr'] = thr_best
                    score['accuracy'] = score_all['accuracy']
                    score['kappa'] = cohen_kappa_score(y_test, y_test_pred)

                    res.iloc[i_row] = pd.Series(score)[col_res]
                    i_row += 1

        res[['ratio',
             'i_cv',
             'thr',
             'support',
             'precision',
             'recall',
             'f1-score',
             'accuracy',
             'kappa']] = res[['ratio',
                              'i_cv',
                              'thr',
                              'support',
                              'precision',
                              'recall',
                              'f1-score',
                              'accuracy',
                              'kappa']].astype(float)

        return res
