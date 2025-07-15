# pip install pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def get_data():
    """生成聚类数据"""
    from sklearn.datasets import make_blobs
    x_value, y_value = make_blobs(n_samples=1000, n_features=40, centers=3, )
    return x_value, y_value


def plot_xy(x_values, label, title):
    """绘图"""
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    df['label'] = label
    sns.scatterplot(x="x", y="y", hue="label", data=df)
    plt.title(title)
    plt.show()


def main():
    x_value, y_value = get_data()
    # PCA 降维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_value)
    plot_xy(x_pca, y_value, "PCA")
    # t-sne 降维
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_value)
    plot_xy(x_tsne, y_value, "t-sne")


if __name__ == '__main__':
    main()