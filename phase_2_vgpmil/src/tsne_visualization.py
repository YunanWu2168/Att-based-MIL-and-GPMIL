import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_and_save(features, instance_labels, config):
    features_embedded = TSNE(n_components=2).fit_transform(features)
    id_0 = np.where(instance_labels == 0)
    id_1 = np.where(instance_labels == 1)

    features_embedded_id_0 = features_embedded[id_0]
    features_embedded_id_1 = features_embedded[id_1]

    plt.scatter(features_embedded_id_0[:,0], features_embedded_id_0[:,1], c="g", alpha=0.5, marker='o',
                label="normal")
    plt.scatter(features_embedded_id_1[:,0], features_embedded_id_1[:,1], c="r", alpha=0.5, marker='+',
                label="ICH")
    #plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    # plt.show()
    print('Save TSNE plot to: ')
    out_name = os.path.basename(config['path_test_df']).split('.')[0]
    out_file = os.path.join(config['output_path'], 'tsne_' + out_name + '.jpg')
    os.makedirs(config['output_path'], exist_ok=True)
    print('Save TSNE plot to: ' + out_file)
    fig.savefig(out_file)