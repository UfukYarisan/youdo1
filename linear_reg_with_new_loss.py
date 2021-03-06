import numpy as np
import sklearn.metrics
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from math import sqrt

def loss_function_for_chart(y_minus_y_pred, error_threshold=8, epsilon = 0.0001):
    if (y_minus_y_pred >= error_threshold):
        return ((epsilon * y_minus_y_pred) + (error_threshold - (error_threshold * epsilon))**2)
    elif (y_minus_y_pred <= -error_threshold):
        return ((-epsilon * y_minus_y_pred) + (error_threshold - (error_threshold * epsilon))**2)
    else:
        return y_minus_y_pred ** 2


def main(verbosity=False):
    st.header("Regression with New Loss Function and L2 Regularization Option")
    st.markdown("""With new loss function, every thing beyond an error treshhold threated as same.    
    **error_treshhold:** The limit value of "y - y_pred" to be threated as same value.  
    **epsilon:** ± slope of piecewise loss function beyond the error treshhold.  
    **lam:** Regularization constant of beta values.  
    **alpha:** Learning rate.  
    **n_max_iter:** Maximum iteration number.  
    """)

    st.header("Dataset")
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target))

    st.dataframe(df)

    st.subheader("House Age independent General Model with OLS")
    fig = px.scatter(df, x="MedInc", y="Price", trendline="ols")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Formulating the new loss function")

    st.markdown("#### General Model")
    st.latex(r"\hat{y}_i=\beta_0 + \beta_1 x_i")

    st.markdown("#### Current Loss Function")

    st.write("Final prediction is a combination (mixtures) of two models")

    st.latex(
        r"-\theta < L < \theta \rightarrow L(\beta_0,\beta_1)=\sum_{i=1}^{N}{(y_i - \hat{y}_i )^2 + \lambda (\beta_0^2 + \beta_1^2)}")
    st.latex(
        r"\theta < L  \rightarrow L(\beta_0,\beta_1)=\sum_{i=1}^{N}{\epsilon(y_i - \hat{y}_i ) + \theta(1-\epsilon) + \lambda (\beta_0^2 + \beta_1^2)}")
    st.latex(
        r"L < -\theta\rightarrow L(\beta_0,\beta_1)=\sum_{i=1}^{N}{-\epsilon(y_i - \hat{y}_i ) + \theta(1-\epsilon) + \lambda (\beta_0^2 + \beta_1^2)}")

    y_minus_y_pred_for_chart = np.arange(-10., 10., 0.1)
    loss_for_chart = []
    for i in range(len(y_minus_y_pred_for_chart)):
        loss_for_chart.append(loss_function_for_chart(y_minus_y_pred_for_chart[i]))

    fig = px.scatter(x=y_minus_y_pred_for_chart.tolist(), y=loss_for_chart)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Almost convex. It will work.")

    st.markdown("#### Partial derivatives")
    st.write("When loss is between minus theta and plus theta")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1)}{\partial \beta_0}=-\sum_{i=1}^{N}{2(y_i - \hat{y}_i) + 2\lambda\beta_0 }")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,)}{\partial \beta_1}=-\sum_{i=1}^{N}{2(y_i - \hat{y}_i)x_i  + 2\lambda\beta_1  }")

    st.write("When loss is biger than plus theta")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1)}{\partial \beta_0}=-\sum_{i=1}^{N}{\epsilon + 2\lambda\beta_0 }")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,)}{\partial \beta_1}=-\sum_{i=1}^{N}{\epsilon  + 2\lambda\beta_1  }")

    st.write("When loss is smaller than minus theta")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1)}{\partial \beta_0}=\sum_{i=1}^{N}{\epsilon + 2\lambda\beta_0 }")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,)}{\partial \beta_1}=\sum_{i=1}^{N}{\epsilon  + 2\lambda\beta_1  }")

    error_threshold = st.slider("Error Treshold (y - y_pred) or (theta)", 1., 10., value=5.)
    epsilon = st.slider("Slope of the lines above the error treshold", 0.000001, 0.1, value=0.0001)
    lam = st.slider("Regularization Multiplier for L2 beta (lam)", 0.001, 1., value=0.1)
    alpha = st.slider("Learning rate (alpha)", 0.000_001, 0.0001, value=0.000_01)
    n_max_iter = st.slider("Maximum iteration number", 100, 10_000_000, value=2000)
    beta, error, mse = reg(df['MedInc'].values, df['Price'].values, error_threshold=error_threshold, epsilon=epsilon, lam=lam,
                      alpha=alpha, n_max_iter=n_max_iter, verbose=verbosity)

    st.latex(fr"Price = {beta[0]:.4f} + {beta[1]:.4f} \times MedInc")
    st.write(f"Error of new loss: {error}")
    st.write(f"Error in terms of MSE: {mse}")

    med_inc = np.linspace(0.0, 15.0, num=150)
    price = np.zeros(len(med_inc))

    for i in range(len(med_inc)):
        price[i] = beta[0] + beta[1] * med_inc[i]
        dict2 = {'MedInc': med_inc[i], 'Price': price[i]}

        df = df.append(dict2, ignore_index=True)
    fig = px.scatter(df, x="MedInc", y="Price", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)
    st.write("SLOPE DIFFERENCE OF OLS AND NEW LOSS CAN BE SEEN ABOVE")


def reg(x, y, error_threshold, epsilon, lam, alpha, n_max_iter, verbose=False):
    beta = np.random.random(2)

    if verbose:
        st.write(beta)
        st.write(x)

    my_bar = st.progress(0.)
    prev_err = 999_999_999
    prev_mse = 0
    prev_beta = []

    for it in range(n_max_iter):

        MSE = 0
        err = 0

        for _x, _y in zip(x, y):
            y_pred = (beta[0] + beta[1] * _x)

            if ((_y - y_pred) >= error_threshold):
                g_b0 = -epsilon + (2 * lam * beta[0])
                g_b1 = -epsilon + (2 * lam * beta[1])

            elif ((_y - y_pred) <= -error_threshold):

                g_b0 = epsilon + (2 * lam * beta[0])
                g_b1 = epsilon + (2 * lam * beta[1])

            else:
                g_b0 = -2 * (_y - y_pred) + (2 * lam * beta[0])
                g_b1 = -2 * ((_y - y_pred) * _x) + (2 * lam * beta[1])

                # st.write(f"Gradient of beta0: {g_b0}")


            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1
            MSE += sqrt((_y - y_pred) ** 2) / len(y)
            if (_y - y_pred) >= error_threshold:
                err += epsilon * (_y - y_pred) + (error_threshold**2 - (error_threshold * epsilon))
            elif (_y - y_pred) <= -error_threshold:
                err += -epsilon * (_y - y_pred) + (error_threshold**2 - (error_threshold * epsilon))
            else:
                err += (_y - y_pred) ** 2


        print(f"{it} - Beta: {beta}, Error: {err}, MSE: {MSE} PREV_ERR: {prev_err}")
        if(err > prev_err):
            print(f"Limit is reached.")
            break
        prev_err = err
        prev_mse = MSE
        prev_beta = beta

        my_bar.progress(it / n_max_iter)

    return prev_beta, prev_err, prev_mse


if __name__ == '__main__':
    main(st.sidebar.checkbox("verbosity"))
