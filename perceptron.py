import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
  
def prepare_dataset(dataset, column_x, column_y, test_size):
   y = dataset[column_x]
   y = np.where(y == column_y, -1, 1)
   x = dataset.drop(column_x, axis=1)
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
 
   return [x_train, x_test, y_train, y_test]
 
 
def testa_registro(row, parameters, bias):
   soma_valor = 0
   for index in range(3):
       soma_valor += row[index] * parameters[index]
   soma_valor += 1*bias
   return soma_valor
 
 
def ativacao(u, t):
   return (1 if u > t else -1)
 
 
# return parameters
def perceptron(X_train, y_train, threshold, eta, epochs):
   # parametros
   b = 0
   w = np.zeros(X_train.shape[1])
 
   for i in np.arange(epochs):
       j = np.random.randint(X_train.shape[0])
       u = np.sum(w*X_train.iloc[j, ]) + b
       y_hat = ativacao(u, threshold)
       if y_train[j] != y_hat:
           w = w + eta*(y_train[j] - y_hat)*X_train.iloc[j, ]
           b = b + eta*(y_train[j] - y_hat)
 
   return [w, b]
 
 
def validate_perceptron(x_test, y_test, parameters, bias):
   counter = 0
   output_array = []
   correct_predictions = 0
  
   for index, row in x_test.iterrows():
       test_result = ativacao(testa_registro(row, parameters, bias), 0.5)

       if test_result == y_test[counter]:
           correct_predictions = correct_predictions + 1

       counter = counter + 1
       output_array.append(test_result)
  
   return {
       'predictions': output_array, 
       'prediction_success': str((correct_predictions / len(y_test) * 100)) + '%'
   }
 

def run_perceptron_validation(dataset, column_x, column_y, test_size):
    x_train, x_test, y_train, y_test = prepare_dataset(dataset, column_x, column_y, test_size)
    
    parameters, bias = perceptron(x_train, y_train, 0.5, 1, 100)
    
    result_validation = validate_perceptron(x_test, y_test, parameters, bias)

    return result_validation
 

 
 
 
 
 
 

