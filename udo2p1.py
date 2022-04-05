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
epsilon = 0.1

# These are our beta vectors. We have approx. 900 users and 1500 items.
beta_user = np.random.rand(r.shape[0])
beta_item = np.random.rand(r.shape[1])

# Old loss value is set to a big number.
old_loss = 999_999_999

# It is going to run until break condition occurs.
while(True):

    # Loss is going to be calculated in each loop.
    loss = 0

    # It is going to be calculated by traversing each cells. (99K cells)
    for index in range(qrow.size):
        r_pred = beta_user[qrow[index]] + beta_item[wcol[index]]
        r_real = r_copy[qrow[index]][wcol[index]]
        loss +=  (r_real - r_pred) ** 2
    loss = loss / 2
    print("LOSS: ", loss)
    print("OLD LOSS: ", old_loss)
    # If not much is learned stop!
    if(old_loss - loss < epsilon):
        break
    old_loss = loss

    # Gradients are calculated for each users and each items.
    g_beta_user = np.zeros(r.shape[0])
    g_beta_item = np.zeros(r.shape[1])
    for index in range(qrow.size):
        row = qrow[index]
        col = wcol[index]
        update = beta_user[row] + beta_item[col] - r_copy[row][col]
        g_beta_user[row] += update
        g_beta_item[col] += update

    # Gradients are updated.
    beta_user = beta_user - g_beta_user * alpha
    beta_item = beta_item - g_beta_item * alpha

print(beta_user)
print("*****")
print(beta_item)
