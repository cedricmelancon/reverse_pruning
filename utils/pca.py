from sklearn.decomposition import PCA
import numpy as np


class PCA_Activation:
    def __init__(self):
        pass

    def run(self, activations_collect, key_idx, components, threshold=0.999):
        """threshold for minimal loss in performance=0.999
            activations_collect  function gathers activations over enough mini batches.
            components=number of filters in the layer you are compressing
            This is for a layer, you need to run this for multiple layers and store optimal_num_filters into a vector
            This vector is the significant dimensionality of all layers"""

        print('number of components are', components)
        activations = activations_collect[key_idx]
        activations = activations.numpy()

        print('shape of activations are:', activations.shape)
        a = activations.swapaxes(1, 2).swapaxes(2, 3)
        a_shape = a.shape
        print('reshaped ativations are of shape', a.shape)

        # number of components should be equal to the number of filters
        pca = PCA(n_components=components)

        # this should be N*H*W,M
        pca.fit(a.reshape(a_shape[0] * a_shape[1] * a_shape[2], a_shape[3]))
        a_trans = pca.transform(
            a.reshape(a_shape[0] * a_shape[1] * a_shape[2], a_shape[3]))
        print('explained variance ratio is:', pca.explained_variance_ratio_)

        np.savetxt('./PCA_files_' + str(key_idx) + '.out',
                      np.cumsum(pca.explained_variance_ratio_))

        optimal_num_filters = np.sum(np.cumsum(pca.explained_variance_ratio_) < threshold)

        print('we want to retain this percentage of explained variance', threshold)
        print('number of filters required to explain that much variance is', optimal_num_filters)

        return optimal_num_filters, pca.components_
