import pandas as pd
import seaborn as sns


def split_stratified(df, feature_by, test_frac=0.25, random_state=None, shuffle=True, visualize=False):
    """
    Function partitions a data-set into train set and test set in a stratified fashion - keeping the same proportion of
    labels as in the original data.
    :param df: pandas dataframe to split into train and test.
    :param feature_by: Column name or number to split according in the staritified fashion. Dataframe must be able to
                       access it in the way: df[feature_by] .
    :param test_frac: Fraction of the data to allocate to test. Train set size is automatically 1 - test_frac.
    :param random_state: Random state to seed the random generator.
    :param shuffle: Shuffle the train and test after splitting (to shuffle if data is sorted in some way). Default is
                    True.
    :param visualize: Visualize the startified split by bar-plotting the amount of each samples from each label in test,
                      train and in total. Using seaborn as sns.
    :return: Two dataframes, train and test, holding together 100% of the samples in df, split in stratified way.
    """
    test = df.index.drop(df.index)
    train = df.index.drop(df.index)

    for label in df[feature_by].unique():
        cur_df = df[df[feature_by] == label]
        test_sample = cur_df.sample(frac=test_frac, axis=0, random_state=random_state).index
        test = test.append(test_sample)
        train = train.append(cur_df.index.drop(test_sample))

    test = df.loc[test]
    train = df.loc[train]
    if shuffle:
        test = test.sample(frac=1)  # shuffle rows due to the append
        train = train.sample(frac=1)

    if visualize:
        # visualize the same proportions:
        test[feature_by].value_counts().plot(kind="bar", alpha=0.4)
        train[feature_by].value_counts().plot(kind="bar", alpha=0.4)
        df[feature_by].value_counts().plot(kind="bar", alpha=0.4)
        sns.plt.title("Labels Proportion in Data Partitioning")
    return train, test
