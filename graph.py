import os.path as op
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load data
rootdir = 'C:/Users/yvonn/Dropbox/work/McGill_PhD/flavor_project/'
reject = [4002, 4007]

df_flav = pd.read_csv(op.join(rootdir, 'data/data_all_run_flavor_condition.csv'), sep=';')
df_flav = df_flav.loc[~df_flav.ID.isin(reject), :]  # reject bad subj

df_money = pd.read_csv(op.join(rootdir, 'data/data_all_run_money_condition.csv'), sep=';')
df_money = df_money.loc[~df_money.ID.isin(reject), :]  # reject bad subj


# clean data
def clean_df(orig_df, r_type):
    orig_df = orig_df.filter(regex=r'(ID|condition|Run|RT|choice)')
    orig_df.columns = orig_df.columns.str.lower()
    orig_df = orig_df.replace({'condition': {'s': 'maltodextrin', 'p': 'placebo'}})

    if r_type == 'flavour':
        bev = {1: 'Neutral', 3: 'CS+', 5: 'CS-'}  # {1:'neutral',3:'paired',5:'unpaired'}
    elif r_type == 'money':
        bev = {1: '$0', 3: '$0.50', 5: '$0.25'}
    else:
        raise ValueError('r_type must be either "flavour" or "money"')
        return

    df_long = pd.DataFrame()
    for bi in bev:

        mx = ['img_' + str(bi) + '_choice', 'img_' + str(bi) + '_rt']
        tmp = orig_df[['id', 'condition', 'run'] + mx]

        for m in ['choice', 'rt']:
            key = [k for k in tmp.columns if m in k]
            tmp = tmp.rename(columns={key[0]: m})

        tmp.loc[:, 'bev'] = np.tile(bev[bi], len(tmp))

        df_long = df_long.append(tmp, ignore_index=True)

    return df_long


def plot_style(r_type):
    colors = {'maltodextrin': 'Reds_r', 'placebo': 'Blues_r'}
    if r_type == 'flavour':
        order = ['CS+', 'CS-', 'Neutral']  # ['paired','unpaired','neutral']
    elif r_type == 'money':
        order = ['$0.50', '$0.25', '$0']

    return colors, order


def plot_choiceprob(df_long, r_type, save=True):
    colors, order = plot_style(r_type)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ci, c in enumerate(['maltodextrin', 'placebo']):
        sns.lineplot(data=df_long[df_long.condition == c],
                     x='run', y='choice',
                     palette=colors[c], hue='bev', hue_order=order,
                     markers=True, dashes=False, ci=95, err_style='bars', err_kws={'capsize': 5, 'capthick': 2},
                     ax=axes[ci])

        axes[ci].set_ylabel('Choice Probability (%)', fontweight='bold');
        axes[ci].set_ylim(0, 1)
        axes[ci].set_xticks([1, 2, 3]);
        axes[ci].set_xticklabels(['Run 1', 'Run 2', 'Run 3'])
        axes[ci].set_xlabel('')
        axes[ci].legend(loc='best')
        axes[ci].set_title(c.capitalize(), fontweight='bold', fontsize=14)

    plt.tight_layout()
    sns.despine()
    if save == True:
        plt.savefig(op.join(rootdir, 'figures', 'run_choiceprob_' + r_type), dpi=600)


def plot_rt(df_long, r_type, save=True):
    colors, order = plot_style(r_type)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ci, c in enumerate(['maltodextrin', 'placebo']):
        sns.barplot(data=df_long[df_long.condition == c],
                    x='run', y='rt',
                    palette=colors[c], hue='bev', hue_order=order,
                    ci=95, ax=axes[ci],
                    ).set_title(c.capitalize(), fontweight='bold', fontsize=14)
        axes[ci].set_ylabel('RT (sec)', fontweight='bold');
        axes[ci].set_ylim(.5, 1.)
        axes[ci].set_xticks([0, 1, 2]);
        axes[ci].set_xticklabels(['Run 1', 'Run 2', 'Run 3'])
        axes[ci].set_xlabel('')
        axes[ci].legend(loc='lower left', framealpha=.95)

    plt.tight_layout()
    sns.despine()
    if save == True:
        plt.savefig(op.join(rootdir, 'figures', 'run_rt_' + r_type), dpi=600)


r_type = {'flavour': df_flav, 'money': df_money}
for r in r_type:
    print('----------' + r.upper() + '----------')
    df = clean_df(r_type[r], r)
    plot_choiceprob(df, r)
    plot_rt(df, r)
----------FLAVOUR - ---------
----------MONEY - ---------

# sort data
df_mean = df.groupby(['id', 'condition', 'bev']).mean()
df_mean.drop('run', axis='columns', inplace=True)
df_mean.reset_index(inplace=True)

# draw
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = ["crimson", "cornflowerblue"]
kwargs = {'hue': 'condition', 'hue_order': ['maltodextrin', 'placebo'],
          'order': ['$0.50', '$0.25', '$0'],
          'palette': colors}
sns.boxplot(data=df_mean,
            x='bev', y='choice', **kwargs,
            ax=axes[0])

axes[0].set_ylabel('Choice Probability of Selecting High Reward (%)', fontweight='bold')
axes[0].set_xlabel('Image Selected', fontweight='bold')
axes[0].get_legend().remove()

sns.barplot(data=df_mean,
            x='bev', y='rt', ci=95, **kwargs,
            ax=axes[1])
axes[1].set_ylim(.6, 1.)
axes[1].set_ylabel('RT (sec)', fontweight='bold')
axes[1].set_xlabel('Image Selected', fontweight='bold')
handles, labels = axes[1].get_legend_handles_labels()
labels = [l.capitalize() for l in labels]
axes[1].legend(handles=handles, labels=labels)

sns.despine()
plt.savefig(op.join(rootdir, 'figures/learning_split_money.png'), dpi=600)
