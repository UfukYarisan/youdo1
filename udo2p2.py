
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
epsilon = 0.25
beta_user = np.random.rand(r.shape[0])
beta_item = np.random.rand(r.shape[1])
old_loss = 999_999_999
lamda_range = np.arange(0, .11, 0.01)
lamda = lamda_range[0]

lamda_loss_dict = {}

for lamda in lamda_range:

    while(True):

        loss = 0
        for index in range(qrow.size):
            row = qrow[index]
            col = wcol[index]
            r_pred = beta_user[row] + beta_item[col]
            r_real = r_copy[row][col]
            loss +=  (r_real - r_pred) ** 2
        loss = loss / 2

        beta_user_square_sum = 0
        for index in range(len(beta_user)):
            beta_user_square_sum += beta_user[index] ** 2
        beta_user_square_sum = (beta_user_square_sum * lamda) / 2

        loss += beta_user_square_sum

        beta_item_square_sum = 0
        for index in range(len(beta_item)):
            beta_item_square_sum += beta_item[index] ** 2
        beta_item_square_sum = (beta_item_square_sum * lamda) / 2

        loss += beta_item_square_sum

        if(old_loss - loss < epsilon):
            lamda_loss_dict[lamda] = loss
            break
        old_loss = loss
        g_beta_user = np.zeros(r.shape[0])
        g_beta_item = np.zeros(r.shape[1])
        for index in range(qrow.size):
            row = qrow[index]
            col = wcol[index]
            g_beta_user_update = beta_user[row] + beta_item[col] - r_copy[row][col]
            g_beta_item_update = beta_user[row] + beta_item[col] - r_copy[row][col]
            g_beta_user[row] += g_beta_user_update
            g_beta_item[col] += g_beta_user_update
        for index in range(len(beta_user)):
            g_beta_user_update = lamda * beta_user[index]
            g_beta_user[index] += g_beta_user_update

        for index in range(len(beta_item)):
            g_beta_item_update = lamda * beta_item[index]
            g_beta_item[index] += g_beta_item_update


        beta_user = beta_user - g_beta_user * alpha
        beta_item = beta_item - g_beta_item * alpha
    loss = 0


    for index in range(test_irow.size):
        row = test_irow[index]
        col = test_jcol[index]
        r_pred = beta_user[row] + beta_item[col]
        r_real = r[row][col]
        loss += (r_real - r_pred) ** 2
    loss = loss / 2

    beta_user_square_sum = 0
    for index in range(len(beta_user)):
        beta_user_square_sum += beta_user[index] ** 2
    beta_user_square_sum = (beta_user_square_sum * lamda) / 2

    loss += beta_user_square_sum

    beta_item_square_sum = 0
    for index in range(len(beta_item)):
        beta_item_square_sum += beta_item[index] ** 2
    beta_item_square_sum = (beta_item_square_sum * lamda) / 2

    loss += beta_item_square_sum
    lamda_loss_dict[lamda] = loss
    print(lamda," is DONE!")

for i in lamda_loss_dict:
    print("LAMDA: ",i," LOSS: ", lamda_loss_dict[i])



