def plot_heat_2d(trans, model, decision_dict, title=None):
    '''
    :param trans: a build data set object
    :param model: an xgboost model
    :param decision_dict: {colums:[range]}
    :param title: plot tile
    :return:
    '''
    import gc
    import xgboost as xgb
    import matplotlib.pyplot as plt

    col_keys = list(decision_dict.keys())
    col_vals = list(decision_dict.values())

    decis_mat = trans.decisionBoundMat(decision_dict)
    decis_features = xgb.DMatrix(decis_mat)
    heat_preds = model.predict(decis_features)
    heat_preds = heat_preds.reshape(len(col_vals[1]), (len(col_vals[0])))


    fig, axis = plt.subplots()
    heatmap = plt.pcolor(heat_preds, cmap=plt.cm.coolwarm)
    plt.colorbar(heatmap)

    plt.ylim(0, max(col_vals[1]) - min(col_vals[1]))
    plt.xlim(0, max(col_vals[0]) - min(col_vals[0]))
    labels_y = [item.get_text() for item in axis.get_xticklabels()]
    labels_x = [item.get_text() for item in axis.get_yticklabels()]
    axis.set_xticklabels([str(i) for i in col_vals[0][0:(len(col_vals[0]) + 1):int(
        len(col_vals[0]) / (len(labels_y) - 1))]])
    axis.set_yticklabels([str(i) for i in col_vals[1][0:(len(col_vals[1]) + 1):int(
        len(col_vals[1]) / (len(labels_x) - 1))]])

    plt.xlabel('Values of ' + col_keys[0])
    # plt.xticks(np.arange(min(key2_seq),max(key2_seq),(max(key2_seq)-min(key2_seq))/10))
    plt.ylabel('Values of ' + col_keys[1])

    if title is not None:
        plt.title(title)
    gc.collect()


def plot_sens_univariate(trans, model, decision_dict, title=None):
    '''
    :param trans: a build data set object
    :param model: an xgboost model
    :param decision_dict: {colums:[range]}
    :param title: plot tile
    :return:
    '''
    import gc
    import xgboost as xgb
    import matplotlib.pyplot as plt

    col_keys = list(decision_dict.keys())
    col_vals = list(decision_dict.values())

    plt.rcParams["figure.figsize"] = 16, 8
    if len(col_keys) == 1:
        decis_mat = trans.decisionBoundMat(decision_dict)
        decis_features = xgb.DMatrix(decis_mat)
        heat_preds = model.predict(decis_features)
        heat_preds = heat_preds.reshape((len(col_vals[0])))

        plt.bar(col_vals[0], heat_preds)
        plt.ylim([min(heat_preds) - .005, max(heat_preds) + .005])
        plt.xlabel('Values of ' + col_keys[0])
        plt.ylabel('Probability')
        plt.title(title)
        gc.collect()
    else:
        for i in range(len(col_keys)):
            decis_mat = trans.decisionBoundMat({col_keys[i]: col_vals[i]})
            decis_features = xgb.DMatrix(decis_mat)
            heat_preds = model.predict(decis_features)
            heat_preds = heat_preds.reshape((len(col_vals[i])))

            plt.subplot(len(col_keys),1,i+1)
            plt.bar(col_vals[i], heat_preds)
            plt.ylim([min(heat_preds) - .005, max(heat_preds) + .005])
            plt.xlabel('Values of ' + col_keys[i])
            plt.ylabel('Probability')
            if i==0:
                plt.title(title)
            gc.collect()

    plt.tight_layout()
    plt.show()

def plot_shap_univar(col, shap_values, features,logged_col=False, feature_names=None,cmap ='RdYlBu_r',xlim=None,
                    dot_size=16, alpha=1, title=None, show=True,plot_horiz=True):
    """
    Create a SHAP dependence plot, colored by an interaction feature.
    Parameters
    ----------
    col : Int or String associated with the column name
        Index of the feature to plot.
    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features)
    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features)
    feature_names : list
        Names of the features (length # features)
    cmap : String/cmap value
        cmap to use in the plots
    xlim : tuple(x1,x2)
        x-limits to use in the plot

    """
    import matplotlib.pyplot as plt
    import gc
    import numpy as np
    # convert from DataFrame
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        features = features.as_matrix()

    def convert_name(col):
        if type(col) == str:
            nzinds = np.where(feature_names == col)[0]
            if len(nzinds) == 0:
                print("Could not find feature named: "+col)
                return None
            else:
                return nzinds[0]
        else:
            return col
    col = convert_name(col)

    # get both the raw and display feature values
    if logged_col==True:
        feature_vals = np.exp(features[:,col])
    else:
        feature_vals = features[:, col]
    shap_vals = shap_values[:,col]
    clow = np.nanpercentile(shap_values[:,col], 2)
    chigh = np.nanpercentile(shap_values[:, col], 98)
    if abs(clow)<abs(chigh):
        clow = -chigh
    else:
        chigh = -clow

    feature_name = feature_names[col]

    # the actual scatter plot
    plt.scatter(feature_vals, shap_vals, s=dot_size, linewidth=0, c=shap_vals,cmap=cmap,vmin=clow, vmax=chigh,
               alpha=alpha, rasterized=len(feature_vals) > 500,edgecolors='k', lw=.2)
    #plot colorbar
    plt.colorbar()
    plt.gcf().set_size_inches(6, 5)
    plt.xlabel(feature_name, fontsize=13)
    plt.ylabel("SHAP value for\n"+feature_name, fontsize=13)
    if plot_horiz==True:#plot horizontal line @ y=0
        plt.axhline(y=0.0, color='k', linestyle='-')
    if title != None:
        plt.title(title, fontsize=13)
    if xlim != None:
        plt.xlim(xlim[0],xlim[1])

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().tick_params( labelsize=11)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#333333")
    if show:
        plt.show()
    gc.collect()




