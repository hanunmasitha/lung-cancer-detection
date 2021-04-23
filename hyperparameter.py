from keras.wrappers.scikit_learn import KerasClassifier

# Define the range of values

# Model Design Components
activation_function = ['relu', 'elu']  # You can also try , 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'
kernel_initializer = ['glorot_uniform', 'he_uniform']  # You can also try , lecun_uniform', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'
optimizer = ['Adam', 'RMSprop']  # You can also try , 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax'

# Hyperparameters
epochs = [10]  # You can also try 20, 30, 40, etc...
batch_size = [4, 16, 32]  # You can also try 2, 4, 8, 16, 32, 64, 128 etc...
dropout_rate = [0.1, 0.2, 0.4]  # No dropout, but you can also try 0.1, 0.2 etc...
learning_rate = [0.1, 0.01, 0.001]
momentum_rate = [0.9, 0.7, 0.5]

def GridSearch(train_set, getModel):
    X_train, Y_train = train_set.next()
    model = KerasClassifier(build_fn=getModel)

    param_grid = dict(epochs=epochs,
                      batch_size=batch_size,
                      optimizer=optimizer,
                      dropout_rate=dropout_rate,
                      #learning_rate = learning_rate,
                      #momentum_rate = momentum_rate,
                      activation_function=activation_function,
                      kernel_initializer=kernel_initializer
                    )

    from sklearn.model_selection import GridSearchCV

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5, verbose=2)
    grid_result = grid.fit(X_train, Y_train)

    # Show results
    best_score = grid_result.best_score_
    best_params = grid_result.best_params_
    print("Best: %f using %s" % (best_score, best_params))

    return best_params

def RandomSearch(train_set, getModel):
    X_train, Y_train = train_set.next()
    model = KerasClassifier(build_fn=getModel)

    param_grid = dict(epochs=epochs,
                      batch_size=batch_size,
                      optimizer=optimizer,
                      dropout_rate=dropout_rate,
                      learning_rate=learning_rate,
                      momentum_rate=momentum_rate,
                      activation_function=activation_function,
                      kernel_initializer=kernel_initializer
    )

    from sklearn.model_selection import RandomizedSearchCV

    n_iter_search = 1
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_grid,
                                       n_iter=n_iter_search,
                                       n_jobs=1, cv=5, verbose=2)
    random_search.fit(X_train, Y_train)

    # Show results
    best_score = random_search.best_score_
    best_params = random_search.best_params_
    print("Best: %f using %s" % (best_score, best_params))

    return best_params