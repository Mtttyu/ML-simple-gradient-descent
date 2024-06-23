The code performs linear regression using gradient descent to predict a target variable based on three features. Here's a breakdown of the code:

1. **Imports:**
   - The code imports necessary libraries such as `random`, `numpy`, `pandas`, and `matplotlib.pyplot`.
   - It also imports specific functions/classes from `sklearn.model_selection` and `sklearn.metrics`.

2. **MSE and MAE Functions:**
   - `MSE` calculates the mean squared error between the predicted values and the actual values.
   - `MAE` calculates the mean absolute error between the predicted values and the actual values.

3. **Gradient Descent Function:**
   - `gradientDescent` performs the gradient descent algorithm to optimize the weights of the linear regression model.
   - It iteratively updates the weights based on the error between the predicted values and the actual values.
   - The function returns the optimized weights and the cost (MSE) at each iteration.

4. **Data Generation:**
   - The code generates synthetic data for training the linear regression model.
   - It creates three features (`X1`, `X2`, `X3`) and calculates the target variable (`Y`) based on these features.
   - The features and the target variable are stored in separate lists (`feature` and `label`).

5. **Data Preprocessing:**
   - The code creates a pandas DataFrame (`df`) using the generated data.
   - The data is then rescaled using mean normalization (subtracting the mean and dividing by the standard deviation).

6. **Splitting Data:**
   - The code splits the rescaled data into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.

7. **Linear Regression:**
   - The code initializes the learning rate (`alpha`) and the number of iterations (`iters`).
   - It initializes the weights for the linear regression model.
   - The `gradientDescent` function is called to optimize the weights using the training data.
   - The predicted values (`y_pred`) are calculated using the optimized weights and the training data.
   - The cost (MSE) of the model is calculated using the training data and the optimized weights.
   - The mean absolute error (MAE) of the model is calculated using the training data and the predicted values.

8. **Output:**
   - The code prints the optimized weights, the mean squared error, and the mean absolute error.