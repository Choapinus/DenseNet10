
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class biometrics(object):
    def __init__(self, mated, non_m):
        self.mated = np.array(mated)
        self.non_m = np.array(non_m)
        self.Nmated = len(mated)
        self.Nnon_m = len(non_m)
        self.gr = (math.sqrt(5) + 1) / 2

    def d_prime(self):
        avg_mated = np.mean(self.mated)
        std_mated = np.std( self.mated)
        avg_non_m = np.mean(self.non_m)
        std_non_m = np.std( self.non_m)
        return np.abs(avg_mated - avg_non_m)/np.sqrt(0.5*(std_mated**2 + std_non_m**2))

    def FAR(self, th):
        return 100*np.sum(self.non_m<th)/self.Nnon_m

    def FARb(self, th):
        return abs(100*np.sum(self.non_m<th)/self.Nnon_m - self.FARv)

    def FRR(self, th):
        return 100*np.sum(self.mated>th)/self.Nmated

    def err(self, th):
        return abs(self.FAR(th) - self.FRR(th))

    def golden_search(self, f, tol=1e-5):
        a = min(self.mated)
        b = max(self.non_m)
        c = b - (b - a) / self.gr
        d = a + (b - a) / self.gr

        while abs(b - a) > tol:
            if f(c) < f(d):
                b = d
            else:
                a = c

            c = b - (b - a) / self.gr
            d = a + (b - a) / self.gr

        return (b + a) / 2

    def get_EER(self):
        EER_th = self.golden_search(f=self.err)
        EER = self.FAR(EER_th)
        return EER, EER_th

    def get_FRR_at(self, FAR_val=0.1):
        self.FARv = FAR_val
        th = self.golden_search(f=self.FARb)
        return self.FAR(th), self.FRR(th), th

    def for_plots(self, d_th = 0.001):
        all_th = np.arange(min(self.mated),max(self.non_m), d_th)
        all_FAR = np.array([self.FAR(th) for th in all_th])
        all_FRR = np.array([self.FRR(th) for th in all_th])
        return all_FAR, all_FRR, all_th

def plot_confusion_matrix2(cm, class_names, figsize=(16, 16), fontsize=36, draw_zeros=True, cmap="YlGnBu_r"):
    sns.set_context("notebook", font_scale=4)
    style = {
        'axes.facecolor': "#EAEAF2",
        'axes.edgecolor': "white",
        'axes.labelcolor': ".15",
        'figure.facecolor': "white",
        'figure.edgecolor': "white",
        'grid.color': "white",
        'grid.linestyle': "-",
        'legend.facecolor': ".8",
        'legend.edgecolor': ".8",
        'text.color': ".15",
        'xtick.color': ".15",
        'ytick.color': ".15",
        'font.family': "sans-serif",
        'font.sans-serif': "Helvetica",
        'savefig.facecolor': "#EAEAF2",
        'savefig.edgecolor': "#white",
        'savefig.transparent': True,
        'eer.color': ".66"
    }
    sns.set_style(style)

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    h, w = cm.shape[:2]
    for i in range(w):
        for j in range(h):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = f"{c}/{s[0]}\n{p:.1f}%"
            elif c == 0 and not draw_zeros:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{c}\n{p:.1f}%"
    cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm.index.name = "Ground Truth"
    cm.columns.name = "Predicted"
    _, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(cm, cmap=cmap, annot=annot, fmt="",
                          annot_kws={'size': fontsize}, ax=ax)
    fig = heatmap.get_figure()
    fig.patch.set_alpha(0)

    return fig