import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns

def f(x):
    return -1.1 * x + 127

def plot_dataset(X,y):

    sns.scatterplot(x=X[:, 0],y=X[:, 1],hue=y)

def plot_boundary(clf, X, y):
    """
    Function to plot a boundary decision
    """
    # define bounds of the domain
    x1min, x1max = X[:, 0].min() - .1, X[:, 0].max() + .1
    x2min, x2max = X[:, 1].min() - .1, X[:, 1].max() + .1
    # define the x and y scale
    x1grid = np.arange(x1min, x1max, 0.01)
    x2grid = np.arange(x2min, x2max, 0.01)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    # make predictions for the grid
    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # reshape the predictions back into a grid
    zz = zz.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Set3')
    
    plt.scatter(X[:, 0], X[:, 1], c= y, s=15, edgecolor='black')


def plot_boundary_pf(clf, pf, X, y):
    """
    Function to plot a boundary decision with polynomial feature transform
    """
    # define bounds of the domain
    x1min, x1max = X[:, 0].min() - .1, X[:, 0].max() + .1
    x2min, x2max = X[:, 1].min() - .1, X[:, 1].max() + .1
    # define the x and y scale
    x1grid = np.arange(x1min, x1max, 0.01)
    x2grid = np.arange(x2min, x2max, 0.01)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    # make predictions for the grid
    zz = clf.predict(pf.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
    # reshape the predictions back into a grid
    zz = zz.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Set3')

    plt.scatter(X[:, 0], X[:, 1], c= y, s=15, edgecolor='black')



def plot_2d_regression_model(reg, dfX, y):
    """
    Function to plot a 2D regression model
    dfX is a pandas dataframe containing 2 columns of attributes
    """
    X1=dfX[dfX.columns[0]]
    X2=dfX[dfX.columns[1]]
    x_range = np.arange(X1.min(), X1.max())
    y_range = np.arange(X2.min(), X2.max())

    xx, yy = np.meshgrid(x_range, y_range)
    # make predictions for the grid
    zz = reg.predict(np.c_[xx.ravel(), xx.ravel()])
    # reshape the predictions back into a grid
    zz = zz.reshape(xx.shape)

    fig = plt.figure(figsize=plt.figaspect(1)*2)
    ax = plt.axes(projection='3d')

    ax.scatter(X1, X2, y, c='r', marker='^')
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha = 0.4)
    ax.set_xlabel(dfX.columns[0])
    ax.set_ylabel(dfX.columns[1])
    ax.set_zlabel('Y')

    plt.show()



def plot_regression_model_pf(reg, pf, dfX, y):
    """
    Function to plot a 1D regression model with polynomial feature transform
    dfX is a pandas dataframe containing 1 column of attributes
    """
    plt.plot(dfX, y, 'ro', markersize=4)
    X=dfX[dfX.columns[0]]
    X_grid = np.arange(X.min(), X.max(), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.plot(X_grid,reg.predict(pf.fit_transform(X_grid)), color = 'b')
    plt.show()



def linearRegressionSummary(model, column_names):
    '''Show a summary of the trained linear regression model'''

    # Plot the coeffients as bars
    fig = plt.figure(figsize=(8,len(column_names)/3))
    fig.suptitle('Linear Regression Coefficients', fontsize=16)
    rects = plt.barh(column_names, model.coef_,color="lightblue")

    # Annotate the bars with the coefficient values
    for rect in rects:
        width = round(rect.get_width(),4)
        plt.gca().annotate('  {}  '.format(width),
                    xy=(0, rect.get_y()),
                    xytext=(0,2),
                    textcoords="offset points",
                    ha='left' if width<0 else 'right', va='bottom')
    plt.show()






def logisticRegressionSummary(model, column_names):
    '''Show a summary of the trained logistic regression model'''

    # Get a list of class names
    numclasses = len(model.classes_)
    if len(model.classes_)==2:
        classes =  [model.classes_[1]] # if we have 2 classes, sklearn only shows one set of coefficients
    else:
        classes = model.classes_

    # Create a plot for each class
    for i,c in enumerate(classes):
        # Plot the coefficients as bars
        fig = plt.figure(figsize=(8,len(column_names)/3))
        fig.suptitle('Logistic Regression Coefficients for Class ' + str(c), fontsize=16)
        rects = plt.barh(column_names, model.coef_[i],color="lightblue")

        # Annotate the bars with the coefficient values
        for rect in rects:
            width = round(rect.get_width(),4)
            plt.gca().annotate('  {}  '.format(width),
                        xy=(0, rect.get_y()),
                        xytext=(0,2),
                        textcoords="offset points",
                        ha='left' if width<0 else 'right', va='bottom')
        plt.show()
        #for pair in zip(X.columns, model_lr.coef_[i]):
        #    print (pair)

def plot_hist_logreg_output(logreg, X):
    wx=logreg.intercept_
    for col in range(logreg.coef_.shape[1]):
        wx=wx+logreg.coef_[0][col]*X[:,col]
    y=logreg.predict_proba(X)

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 5))
    #fig.suptitle('Horizontally stacked subplots')
    ax1.hist(wx, bins = 40, rwidth=0.8);
    ax1.set_xlabel('$w^T x$')
    #ax1.set_ylabel('')
    ax1.grid(alpha =0.3)
    ax1.set_title(r'Histogram of $w^T x$')
    ax2.hist(y[:,1], bins = 40, rwidth=0.8);
    ax2.set_xlabel('$h_w(x)$')
    #ax2.set_ylabel('')
    ax2.set_title(r'Histogram of $h_w(x)$')
    major_ticks = np.arange(0, 1, 10)
    ax2.set_xticks(major_ticks)
    ax2.grid(which='major',alpha =0.3)
    plt.show()



def plotCoeffEvolution(titleText, regParamValues, coeffitients):
    '''Show the evolution of coefficients, for different values of the regularisation parameter'''

    plt.figure()
    ax = plt.gca()
    ax.plot(regParamValues, coeffitients)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title(titleText)
    plt.axis('tight')
    plt.show()


def plotErrorEvolution(titleText, regParamValues, errors):
    '''Show the evolution of errors, for different values of the regularisation parameter'''
    plt.figure()
    ax = plt.gca()
    ax.plot(regParamValues, errors)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('error')
    plt.axis('tight')
    plt.title(titleText)
    plt.show()

def plotMeanScores(alphas, results):
    plt.title('Mean scores as a function of the hyperparameter')
    plt.plot(alphas, results['mean_train_score'], marker = 'o', label='Train')
    plt.plot(alphas, results['mean_test_score'],  marker = 'o',label='Valid')
    plt.legend()
    plt.xlabel('hyper-parameter')
    plt.ylabel('score')
    plt.xticks(alphas)


def plotLassoPredictionError(model):
    plt.figure(figsize=(10, 6), constrained_layout=True)
    plt.semilogx(model.alphas_, model.mse_path_, ':')
    plt.semilogx(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(model.alpha_, linestyle='--', color='k',
            label='alpha: CV estimate')
    print(model.alphas_)

    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Mean square prediction error')
    plt.show(block=False)


def plotRidgePredictionError(alphas, results, alpha):
    plt.figure(figsize=(10, 6), constrained_layout=True)
    plt.semilogx(alphas, results, ':')
    plt.semilogx(alphas, results.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(alpha, linestyle='--', color='k',
            label='alpha: CV estimate')

    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Mean square prediction error')
    plt.show(block=False)
