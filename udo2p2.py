
import pandas as pd
import numpy as np
# Data is pulled into a data frame
df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                 names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')

# Dataframe is pivoted into a matrix
r = df.pivot(index='user_id', columns='item_id', values='rating').values

# NonNA values are found.
# There are 100K different cells.
# r[irow[index]][jcol[index]] reaches the cell you want.
irow, jcol = np.where(~np.isnan(r))

# 1K random indexes are selected to mask.
idx = np.random.choice(np.arange(100_000), 1000, replace=False)
# Below are test cells.
test_irow = irow[idx]
test_jcol = jcol[idx]

# We are gonna work with r_copy and we are gonna mask this!
r_copy = r.copy()

# Masking is done with only to the filled cells.
for index in range(test_irow.size):
    r_copy[test_irow[index]][test_jcol[index]] = np.nan

# Here are our new NonNA cells. There are 99K of them.
qrow, wcol = np.where(~np.isnan(r_copy))

# Alpha is the learning rate.
alpha = 0.001
# Epsilon is used in the stop condition.
epsilon = 0.25

# These are our beta vectors. We have approx. 900 users and 1500 items.
beta_user = np.random.rand(r.shape[0])
beta_item = np.random.rand(r.shape[1])

# Old loss value is set to a big number.
old_loss = 999_999_999

# A range of lambda is created to check which one is good.
# The validation set is going to be used to optimize which one is good.
lamda_range = np.arange(0, .11, 0.01)
lamda = lamda_range[0]

# Hold lambda, loss pairs
lamda_loss_dict = {}

# Runs for each lambda
for lamda in lamda_range:

    # It is going to run until break condition occurs.
    while(True):

        # Loss is going to be calculated in each loop.
        loss = 0

        # It is going to be calculated by traversing each cells. (99K cells)
        for index in range(qrow.size):
            row = qrow[index]
            col = wcol[index]
            r_pred = beta_user[row] + beta_item[col]
            r_real = r_copy[row][col]
            loss +=  (r_real - r_pred) ** 2
        loss = loss / 2

        # New terms are added!!
        beta_user_square_sum = 0
        for index in range(len(beta_user)):
            beta_user_square_sum += beta_user[index] ** 2
        beta_user_square_sum = (beta_user_square_sum * lamda) / 2

        loss += beta_user_square_sum

        # New terms are added!!
        beta_item_square_sum = 0
        for index in range(len(beta_item)):
            beta_item_square_sum += beta_item[index] ** 2
        beta_item_square_sum = (beta_item_square_sum * lamda) / 2

        loss += beta_item_square_sum

        # If not much is learned stop!
        if(old_loss - loss < epsilon):
            lamda_loss_dict[lamda] = loss
            break
        old_loss = loss

        # Gradients are calculated for each users and each items.
        g_beta_user = np.zeros(r.shape[0])
        g_beta_item = np.zeros(r.shape[1])
        for index in range(qrow.size):
            row = qrow[index]
            col = wcol[index]
            g_beta_user_update = beta_user[row] + beta_item[col] - r_copy[row][col]
            g_beta_item_update = beta_user[row] + beta_item[col] - r_copy[row][col]
            g_beta_user[row] += g_beta_user_update
            g_beta_item[col] += g_beta_user_update

        # New terms are added!!
        for index in range(len(beta_user)):
            g_beta_user_update = lamda * beta_user[index]
            g_beta_user[index] += g_beta_user_update

        # New terms are added!!
        for index in range(len(beta_item)):
            g_beta_item_update = lamda * beta_item[index]
            g_beta_item[index] += g_beta_item_update

        # Gradients are updated.
        beta_user = beta_user - g_beta_user * alpha
        beta_item = beta_item - g_beta_item * alpha

    # This loss is the loss that are going to be calculated with validation set for that specific lambda.
    loss = 0

    # For eached masked value loop will run.
    for index in range(test_irow.size):
        row = test_irow[index]
        col = test_jcol[index]
        r_pred = beta_user[row] + beta_item[col]
        r_real = r[row][col]
        loss += (r_real - r_pred) ** 2
    loss = loss / 2

    # New terms are added!!
    beta_user_square_sum = 0
    for index in range(len(beta_user)):
        beta_user_square_sum += beta_user[index] ** 2
    beta_user_square_sum = (beta_user_square_sum * lamda) / 2

    loss += beta_user_square_sum

    # New terms are added!!
    beta_item_square_sum = 0
    for index in range(len(beta_item)):
        beta_item_square_sum += beta_item[index] ** 2
    beta_item_square_sum = (beta_item_square_sum * lamda) / 2

    loss += beta_item_square_sum

    # Added to the dictionary.
    lamda_loss_dict[lamda] = loss
    print(lamda," is DONE!")

# Dictionary is printed to see which is the better.
for i in lamda_loss_dict:
    print("LAMDA: ",i," LOSS: ", lamda_loss_dict[i])



