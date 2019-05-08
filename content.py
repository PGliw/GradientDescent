# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


def sigmoid(x):
    """
    Wylicz wartość funkcji sigmoidalnej dla punktów *x*.

    :param x: wektor wartości *x* do zaaplikowania funkcji sigmoidalnej Nx1
    :return: wektor wartości funkcji sigmoidalnej dla wartości *x* Nx1
    """
    return 1 / (1 + np.exp(-x))


def logistic_cost_function(w, x_train, y_train):
    """
    Wylicz wartość funkcji logistycznej oraz jej gradient po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej,
        a *grad* jej gradient po parametrach *w* Mx1
    """
    # list of sigmoid function values for every x in x_train
    sigmas = sigmoid(x_train @ w)
    N = x_train.shape[0]
    # calculate likelihood
    p_D_w = np.prod([(sigmas[n] if y_train[n] == 1 else (1 - sigmas[n])) for n in range(N)])
    # calculate logarithm of likelihood
    ln_p_D_w = np.log(p_D_w)
    # calculate objective function
    L_w = -ln_p_D_w / N
    # calculate gradient of objective function (lecture 4)
    grad_L_w = -(np.transpose(x_train) @ (y_train - sigmas)) / N
    return L_w, grad_L_w


def gradient_descent(obj_fun, w0, epochs, eta):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą algorytmu gradientu
    prostego, korzystając z kroku uczenia *eta* i zaczynając od parametrów *w0*.
    Wylicz wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość
    parametrów modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argument
        wektor parametrów *w* [wywołanie *val, grad = obj_fun(w)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok algorytmu gradientu prostego
    :param eta: krok uczenia
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu w każdej
        epoce (lista o długości *epochs*)
    """
    w = w0  # initial point w for objective function L
    log_values = []  # list of values of cost function for each epoch
    #  find optimal vector of parameters - w
    for k in range(epochs):
        fun_value, fun_grad = obj_fun(w)
        if k > 0:  # omit initial fun value ( for w0 )
            log_values.append(fun_value)
        w = w - eta * fun_grad  # calculate next w

    # calculate fun value for optimal w
    last_fun_value, _ = obj_fun(w)

    log_values.append(last_fun_value)
    return w, log_values


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą stochastycznego
    algorytmu gradientu prostego, korzystając z kroku uczenia *eta*, paczek
    danych o rozmiarze *mini_batch* i zaczynając od parametrów *w0*. Wylicz
    wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość parametrów
    modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argumenty
        wektor parametrów *w*, paczkę danych składających się z danych
        treningowych *x* i odpowiadających im etykiet *y*
        [wywołanie *val, grad = obj_fun(w, x, y)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu dla całego
        zbioru treningowego w każdej epoce (lista o długości *epochs*)
    """
    w = w0  # initial point w for objective function L
    log_values = []  # list of values of cost function for each epoch
    number_of_batches = int(y_train.shape[0] / mini_batch)
    # split training data into batches
    x_batches, y_batches = np.split(x_train, number_of_batches), np.split(y_train, number_of_batches)

    #  find optimal vector of parameters - w
    for i in range(epochs):
        for x, y in zip(x_batches, y_batches):
            fun_value, fun_grad = obj_fun(w, x, y)
            w = w - eta * fun_grad
        epoch_log_value, _ = obj_fun(w, x_train, y_train)
        log_values.append(epoch_log_value)
    return w, log_values


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """
    Wylicz wartość funkcji logistycznej z regularyzacją l2 oraz jej gradient
    po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param regularization_lambda: parametr regularyzacji l2
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej
        z regularyzacją l2, a *grad* jej gradient po parametrach *w* Mx1
    """
    pass


def prediction(x, w, theta):
    """
    Wylicz wartości predykowanych etykiet dla obserwacji *x*, korzystając
    z modelu o parametrach *w* i progu klasyfikacji *theta*.

    :param x: macierz obserwacji NxM
    :param w: wektor parametrów modelu Mx1
    :param theta: próg klasyfikacji z przedziału [0,1]
    :return: wektor predykowanych etykiet ze zbioru {0, 1} Nx1
    """
    sigmas = sigmoid(x @ w)
    ys = sigmas > theta  # nice python syntax
    return ys


def f_measure(y_true, y_pred):  # to count false positives and false negatives
    """
    Wylicz wartość miary F (F-measure) dla zadanych rzeczywistych etykiet
    *y_true* i odpowiadających im predykowanych etykiet *y_pred*.

    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet predykowanych przed model Nx1
    :return: wartość miary F (F-measure)
    """
    true_positives = np.sum([y_true[n] == 1 and y_pred[n] == 1 for n in range(len(y_true))])
    false_positives = np.sum([y_true[n] == 0 and y_pred[n] == 1 for n in range(len(y_true))])
    false_negatives = np.sum([y_true[n] == 1 and y_pred[n] == 0 for n in range(len(y_true))])
    result = 2 * true_positives / (2 * true_positives + false_negatives + false_positives)
    return result


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    Policz wartość miary F dla wszystkich kombinacji wartości regularyzacji
    *lambda* i progu klasyfikacji *theta. Wyznacz parametry *w* dla modelu
    z regularyzacją l2, który najlepiej generalizuje dane, tj. daje najmniejszy
    błąd na ciągu walidacyjnym.

    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param x_val: zbiór danych walidacyjnych NxM
    :param y_val: etykiety klas dla danych walidacyjnych Nx1
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :param lambdas: lista wartości parametru regularyzacji l2 *lambda*,
        które mają być sprawdzone
    :param thetas: lista wartości progów klasyfikacji *theta*,
        które mają być sprawdzone
    :return: krotka (regularization_lambda, theta, w, F), gdzie
        *regularization_lambda* to wartość regularyzacji *lambda* dla
        najlepszego modelu, *theta* to najlepszy próg klasyfikacji,
        *w* to parametry najlepszego modelu, a *F* to macierz wartości miary F
        dla wszystkich par *(lambda, theta)* #lambda x #theta
    """
    F = []  # matrix of f measures for each pair (lambda, theta)
    parameters_list = []
    for lamb in lambdas:
        w, log_values = stochastic_gradient_descent(lambda w, x, y: regularized_logistic_cost_function(w, x, y, lamb), x_train, y_train, w0, epochs, eta, mini_batch)
        F_lamb = []
        for theta in thetas:
            y_pred = prediction(x_val, w, theta)
            f_lamb_t = f_measure(y_val, y_pred)
            parameters_list.append((lamb, theta, w))
            F_lamb.append(f_lamb_t)
        F.append(F_lamb)

    best_F_index = int(np.argmax(F))
    regularization_lambda, theta, w = parameters_list[best_F_index]
    return regularization_lambda, theta, w, F
