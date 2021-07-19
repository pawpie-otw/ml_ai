import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE


def array_coverter(array):
    """
    convert data into 2 divided cols
    i use it to divide data from files to 2 columns: X and Y
    """
    ar1 = array[:, 0].reshape(len(array), 1)
    ar2 = array[:, 1].reshape(len(array), 1)
    return ar1, ar2


def polynomial_regression(attributes, values, degree):
    '''
    attributes - features vectore (X)
    degree - degree of polynomial
    values - values of features
    '''
    # utworzenie PF odpowiedniego stopnia
    reg = PolynomialFeatures(degree=degree)
    # generowanie potęg <2,N-1> dla atrybutów (Xów)
    polynomial = reg.fit_transform(attributes)
    # utworzenie i nauczanie modelu, czyli próba dopasowania współczynników
    pol_reg = LinearRegression()
    pol_reg.fit(polynomial, values)
    # zwracamy wyuczony model i obiekt PF {degree} stopnia
    return reg, pol_reg


def prepare_regression(train_set, test_set, max_degree):
    '''
    attributes - wektor cech (X), zarówno zbiór treningowy jak i testowy
    values - wartości odpowiadające cechom z 'attributes'
    max_degree - maksymalny stopień wielomianu, przy pomocy którego chcemy wykonać regresję
    '''
    # dzielimy tablice odpowiednio do wymagań
    X_train,  y_train = array_coverter(train_set)
    X_test, y_test = array_coverter(test_set)
    # puste listy na zwracane dane: liste modeli i bledow
    models, test_errors = [], []
    # ustawienei wymiarów planszy na wykresy
    plt.figure(figsize=(10, 12))
    for i in range(2, max_degree + 1):
        poly_model, linear_model = polynomial_regression(X_train, y_train, i)
        # dolaczamy model do listy modeli
        models.append(linear_model)
        # sprawdzamy przewidywane wyniki na zbiorze testowym
        y_test_prediction = linear_model.predict(
            poly_model.fit_transform(X_test))
        # zapisujemy różne między przewidywanymi wynikami a rzeczywistymi ze zbioru testowego
        test_errors.append(MSE(y_test_prediction, y_test))

        # Rysowanie wykresów gęstości błedu
        plt.subplot(5, 4, i-1)
        plt.xlabel('Error values')
        plt.ylabel('Density')
        plt.hist(y_test_prediction-y_test, color='red', density=True)
        # tytuł zawiera stopien i zaokrąglony blad
        plt.title(f"degree: {i}; MSE: {round(test_errors[-1],2)}")

    plt.tight_layout()
    plt.show()
    # zwracamy tylko model, ktory ma najmniejszy MSE
    return models[np.argmin(test_errors)], test_errors[np.argmin(test_errors)]


if __name__ == "__main__":
    pass
