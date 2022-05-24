import numpy as np
import tensorflow as tf

def leave_users_out(full_data, leave_out, seed=1234):
    """
    Leaves leave_out number of users out of the full dataset
    :param full_data: pandas df containing user_id, item_id sorted on datetime per user (ascending)
    :param leave_out: number of users to leave out
    :param seed: set seed for picking the same users
    :return: full_data df without leave_out users and the left out subset of leave_out users
    """
    np.random.seed(seed)
    full_data['index'] = full_data.index
    user_index_df = full_data.groupby('user_id')['index'].apply(list)
    users_indices = []

    if type(leave_out) is list:
        users = leave_out
    else:
        users = np.random.choice(list(user_index_df.index), leave_out, replace=False)

    for user in users:
        users_indices.extend(user_index_df.loc[user])

    sub_set = full_data.loc[users_indices].drop(columns=['index'])
    remaining = full_data.drop(users_indices).drop(columns=['index'])

    return remaining, sub_set


def leave_last_x_out(full_data, n_users, leave_out=1, seqs=False, already_picked=[], seed=1234):
    """
    Leaves last leave_out items out for all n_users in full_data
    :param full_data: pandas df containing user_id, item_id sorted on datetime per user (ascending)
    :param n_users: number of users to leave the last leave_out items out of the full_data
    :param leave_out: number of last item_ids per user
    :param already_picked: If method is used again for the same full_data, provide list of users to not have overlap
    :param seed: set seed for picking the same users
    :return: full_data df without leave_out items per users and the left out subset of leave_out item_id(s)
             per user_id(s)
    """
    np.random.seed(seed)

    full_data['index'] = full_data.index
    user_items_ind = full_data.groupby('user_id')['index'].apply(list)
    users = full_data.user_id.unique()
    leave_out_indices = []
    users_picked = []

    for i in range(len(full_data.user_id.unique())):
        if len(users_picked) >= n_users:
            break

        random_user = np.random.choice(users)
        item_indices = user_items_ind[random_user]  # random user's items indices
        if random_user in users_picked or len(item_indices) <= leave_out or random_user in already_picked:
            random_user = np.random.choice(users)
            item_indices = user_items_ind[random_user]  # random user's items indices
        else:
            users_picked.append(random_user)
            leave_out_indices.extend(item_indices[-leave_out:])

    if len(users_picked) < n_users:
        error = 'Cannot pick ' + str(n_users) + ' users with ' + str(leave_out) + ' items'
        solution = '\nTry a smaller test and/or validation percentage of the data or less items to leave out'
        raise ValueError(error + solution)


    if seqs:
        all_remaining, left_out = leave_users_out(full_data, users_picked)
        remaining = left_out.drop(leave_out_indices)
        left_out = left_out.loc[leave_out_indices]
        if leave_out == 0:
            return [all_remaining, left_out, remaining]
        else:
            return [all_remaining, remaining, left_out]

    # drops last items for n_users users
    leave_out_set = full_data.loc[leave_out_indices].drop(columns=['index'])  # the last items of n_users users with n_item > leave_out
    full_data_leave_one_out = full_data.drop(leave_out_indices).drop(columns=['index'])

    return [full_data_leave_one_out, leave_out_set]


def train_val_test_split(df, val_perc, test_perc, n_items_val, n_items_test, seqs=False, stats=True):
    """
    Create specific train, validation and test set for recommender systems
    :param df: pandas df containing user_id, item_id sorted on datetime per user (ascending)
    :param val_perc: percentage of users to use for validation set
    :param test_perc: percentage of users to use for test set
    :param n_last_items: number of items to leave out of df and put in validation and test set per user
    :param seqs:
    :param stats:
    :return:
    """
    total_users = len(df.user_id.unique())  # Need all users for BPR

    test_users = int(test_perc * total_users)  # Number of users to be used for testing
    val_users = int(val_perc * total_users)

    train_test = leave_last_x_out(df, test_users, n_items_test, seqs=seqs)
    test_users_list = train_test[1].user_id.unique()
    train_val = leave_last_x_out(train_test[0], val_users, n_items_val, already_picked=test_users_list, seqs=seqs)

    # if stats:
    #     print('Total users:', total_users)
    #     print('Number of train users:', len(train_set.user_id.unique()))
    #     print('Number of test users:', test_users)
    #     print('Number of validation users:', val_users, '\n')

    return [*train_val, *train_test[1:]]


def get_x_y_sequences(dataset, shift=1, stats=True):
    """

    :param dataset: pandas df containing user_id, item_id sorted on datetime per user (ascending)
    :param shift: by how much should the target (y) sequence be shifted
    :param ordered: should the sequences be ordered by length
    :param stats: print number of sequences, avg length, std_dev, median
    :return: list user_sequences_x, list shifted user_sequences_y and pandas df user_order if order is True,
             else float median
    """
    user_sequences_x = []
    user_sequences_y = []
    lengths = []

    users = dataset.user_id.unique()

    for u in users:
        user_item_seq = np.array(dataset[dataset['user_id'] == u]['item_id'])
        user_sequences_x.append(user_item_seq[:-shift])
        user_sequences_y.append(user_item_seq[shift:])
        lengths.append(len(user_item_seq))

    median = np.median(lengths)

    if stats:
        print('Number of sequences x:', len(user_sequences_x),
              '\nAvg sequence length x:', np.average(lengths),
              '\nStd_dev sequence length x:', np.round(np.std(lengths), 2),
              '\nMedian of sequence length x:', median)

    return user_sequences_x, user_sequences_y, median


def standard_padding(sequences, max_length, pad_value=0.0, eval=False, stats=True):
    """
    Pads (post) sequences up until max_length with zeros
    :param sequences: list of sequences per user
    :param max_length: maximum length to pad sequences to
    :param eval: whether the padded sequences will be used for evaluation (in get_predictions)
    :param stats: print number of sequences, acg sequence length, st_dev of sequence length
    :return: tensorflow dataset consisting of the padded sequences
    """
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                        sequences, maxlen=int(max_length), padding='post', truncating='pre', value=pad_value)
    if stats:
        print('number of sequences:', padded_sequences.shape[0], 
              '\navg sequence length:', np.average([i.shape[0] for i in padded_sequences]),
              '\nstd_dev sequence length:', np.std([i.shape[0] for i in padded_sequences]))

    if eval:
        return padded_sequences

    return tf.data.Dataset.from_tensor_slices(padded_sequences)



def create_seq_batch_dataset(df, shift, max_seq_len, pad_value, batch_size, stats=True, drop_remainder=True):
    """

    :param df:
    :param shift:
    :param max_seq_len:
    :param pad_value:
    :param batch_size:
    :param drop_remainder:
    :return:
    """
    user_sequences_x, user_sequences_y, median = get_x_y_sequences(df, shift, stats=stats)
    sequences_data_x = standard_padding(user_sequences_x, max_seq_len, pad_value=pad_value, stats=stats)
    sequences_data_y = standard_padding(user_sequences_y, max_seq_len, pad_value=pad_value, stats=stats)

    dataset = tf.data.Dataset.zip((sequences_data_x, sequences_data_y))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset


def split_df_by_users(df, left_out_items, n_splits):
    user_list = df.user_id.unique()
    data_size = len(user_list) / n_splits
    leftovers = 0

    if data_size - int(data_size) > 0:
        leftovers = n_splits * (data_size - int(data_size)) - 1

    split = [n for n in range(0, len(user_list), int(data_size))]
    split[-1] = split[-1] + int(leftovers)
    data_split = [user_list[split[n]:split[n + 1]] for n in range(len(split) - 1)]

    df_splits = []
    left_out_items_split = []
    for users_split in data_split:
        _, df_subset = leave_users_out(df, list(users_split))
        _, subset = leave_users_out(left_out_items, list(users_split))

        df_splits.append(df_subset)
        left_out_items_split.append(subset)

    return df_splits, left_out_items_split


def leave_last_out(df, n_items=1):
    leave_out_indices = []
    for u in df.user_id.unique():
        df_user = df[df.user_id == u]
        leave_out_indices.extend(df_user.iloc[-1:].index)
    leave_out = df.loc[leave_out_indices]
    remaining = df.drop(leave_out_indices)
    
    return remaining, leave_out