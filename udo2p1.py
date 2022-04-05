
import pandas as pd
import numpy as np

df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                 names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')

r = df.pivot(index='user_id', columns='item_id', values='rating').values

irow, jcol = np.where(~np.isnan(r))

idx = np.random.choice(np.arange(100_000), 1000, replace=False)
test_irow = irow[idx]
test_jcol = jcol[idx]

r_copy = r.copy()

for index in range(test_irow.size):
    r_copy[test_irow[index]][test_jcol[index]] = np.nan

qrow, wcol = np.where(~np.isnan(r_copy))

alpha = 0.001
epsilon = 0.1
beta_user = np.random.rand(r.shape[0])
beta_item = np.random.rand(r.shape[1])
old_loss = 999_999_999
while(True):

    loss = 0
    for index in range(qrow.size):
        r_pred = beta_user[qrow[index]] + beta_item[wcol[index]]
        r_real = r_copy[qrow[index]][wcol[index]]
        loss +=  (r_real - r_pred) ** 2
    loss = loss / 2
    print("LOSS: ", loss)
    print("OLD LOSS: ", old_loss)
    if(old_loss - loss < epsilon):
        break
    old_loss = loss
    g_beta_user = np.zeros(r.shape[0])
    g_beta_item = np.zeros(r.shape[1])
    for index in range(qrow.size):
        row = qrow[index]
        col = wcol[index]
        update = beta_user[row] + beta_item[col] - r_copy[row][col]
        g_beta_user[row] += update
        g_beta_item[col] += update

    beta_user = beta_user - g_beta_user * alpha
    beta_item = beta_item - g_beta_item * alpha

print(beta_user)
print("*****")
print(beta_item)
"""
err = []
for u, j in zip(test_irow, test_jcol):
    y_pred = user.predict1(r_copy, u, j)
    y = r[u, j]

    err.append((y_pred - y) ** 2)

cons.print(f"RMSE: {np.sqrt(np.nanmean(np.array(err)))}")

"""
