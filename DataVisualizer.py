from enum import Enum

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from strenum import StrEnum


class DataVisualizer:
    
    class ReductionMethod(Enum):
        PCA = 1
        TSNE = 2

    class PCNames(StrEnum):
        """ Principal Components names, used in dataframe and plotting """
        PC1 = 'pc-one'
        PC2 = 'pc-two'
        PC3 = 'pc-three'

    def __init__(self, data: pd.DataFrame, y_column: str, detectors: pd.DataFrame = None,
                 reduction_method=ReductionMethod.PCA):
        # Set columns names
        self.features_columns = list(data)
        self.features_columns.remove(y_column)
        self.y_column = y_column

        # Reduce dataset to three components
        if reduction_method == self.ReductionMethod.PCA:
            self.reduced_df, self.explained_variance = self.__reduce_with_PCA(data)
        elif reduction_method == self.ReductionMethod.TSNE:
            self.reduced_df, self.explained_variance = self.__reduce_with_tSNE(data)
        self.reduced_df['y'] = data[self.y_column]  # add ground truth to the new dataframe

        # Reduce dataset to three components
        self.detectors_df = None
        if detectors is not None:
            self.detectors_df, _ = self.__reduce_with_PCA(detectors)

    def visualize_2D(self):
        plt.figure(figsize=(16, 10))
        ax = sns.scatterplot(
            x=self.PCNames.PC1, y=self.PCNames.PC2,
            hue='y',
            style='y',
            markers=['X', 'o'],
            palette=['red', 'green'],
            data=self.reduced_df,
            alpha=.8  # trasparenza=0, opaco=1
        )
        if self.detectors_df is not None:
            ax.scatter(
                x=self.detectors_df[self.PCNames.PC1],
                y=self.detectors_df[self.PCNames.PC2],
                c='blue',
                alpha=.1,  # trasparenza=0, opaco=1
                marker='o',
                label='Detector',
                s=5
            )

        ax.set_title('2D Scatter Plot')

        # Inseriamo delle label personalizzate nella legenda
        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels = ['Self', 'Not Self']
        if self.detectors_df is not None:
            labels.insert(0, 'Detector')
        ax.legend(handles, labels)

        # Visualizza il plot
        plt.show()

    def visualize_3D(self):
        ax = plt.figure(figsize=(16, 10)).gca(projection='3d')

        selfs_df = self.reduced_df[self.reduced_df['y'] == 1]
        not_selfs_df = self.reduced_df[self.reduced_df['y'] == 0]
        # print(selfs_df)
        # print(not_selfs_df)

        ax.scatter(
            xs=selfs_df[self.PCNames.PC1],
            ys=selfs_df[self.PCNames.PC2],
            zs=selfs_df[self.PCNames.PC3],
            c='green',
            alpha=.8,  # trasparenza=0, opaco=1
            marker='o',
            label='Self'
        )

        ax.scatter(
            xs=not_selfs_df[self.PCNames.PC1],
            ys=not_selfs_df[self.PCNames.PC2],
            zs=not_selfs_df[self.PCNames.PC3],
            c='red',
            alpha=.8,  # trasparenza=0, opaco=1
            marker='X',
            label='Not Self'
        )

        ax.set_title('3D Scatter Plot')
        ax.set_xlabel(self.PCNames.PC1)
        ax.set_ylabel(self.PCNames.PC2)
        ax.set_zlabel(self.PCNames.PC3)
        ax.legend()

        plt.show()

    def get_total_variance_explained(self):
        return sum(self.explained_variance)

    ###################################################
    # Private methods declaration
    ###################################################

    def __reduce_with_PCA(self, data: pd.DataFrame) -> (pd.DataFrame, ):
        """ Apply PCA to the input dataset and reduce it at three dimensions.

        :param data:
        :return: A DataFrame with the three principal components computed and their explained variation.
        """
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data[self.features_columns].values)

        reduced_data = pd.DataFrame()
        reduced_data[self.PCNames.PC1] = pca_result[:, 0]
        reduced_data[self.PCNames.PC2] = pca_result[:, 1]
        reduced_data[self.PCNames.PC3] = pca_result[:, 2]
        # print(reduced_data)

        return reduced_data, pca.explained_variance_ratio_

    def __reduce_with_tSNE(self, data: pd.DataFrame, verbose: int = 0) -> (pd.DataFrame, ):
        """ Apply t-SNE (a probabilistic reduction method) to the input dataset and reduce it at three dimensions.

        :param data: The dataframe to reduce
        :return: A DataFrame with the three principal components computed and their explained variation.
        """
        tsne = TSNE(n_components=3, verbose=verbose, perplexity=50, n_iter=5000, init='pca')
        tsne_results = tsne.fit_transform(data[self.features_columns].values)

        reduced_data = pd.DataFrame()
        reduced_data[self.PCNames.PC1] = tsne_results[:, 0]
        reduced_data[self.PCNames.PC2] = tsne_results[:, 1]
        reduced_data[self.PCNames.PC3] = tsne_results[:, 2]
        print(reduced_data)

        return reduced_data, None  # TODO: insert explained variance here


if __name__ == '__main__':
    try:
        df = pd.read_csv("data/Alzheimer_reduced_normalized.csv")
        det_df = pd.read_csv("results/2022_06_06-22.20.10-4000dts/Rad_5.03/NsaSeed_9311/ShuffleSeed_2903/self_tolerants_ALC.csv")

        # print(df, "\n")
        # print(det_df, "\n")
        visualizer = DataVisualizer(df, y_column='target', reduction_method=DataVisualizer.ReductionMethod.PCA,
                                    detectors=det_df)

        # print(f'Variance explained by the PCA: {round(visualizer.get_total_variance_explained(), 4)}')
        visualizer.visualize_2D()
        visualizer.visualize_3D()
    except KeyboardInterrupt:
        print(f'Execution stopped for a CTRL+C interrupt.')
