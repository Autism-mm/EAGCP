map_idx:对齐数据索引。√

unmap_idx：不对齐数据索引。√

map_label:对齐数据索引。√

unmap_label:不对齐数据索引。√

map_X：第一个视图对齐的原数据集。√

map_Y：第二个视图对齐的原数据集。√

unmap_X：第一个视图未对齐的原数据集。√

unmap_Y:第二个视图未对齐的原数据集。√

unmaplabel_X：未对齐的第一个视图的标签。√

unmaplabel_Y：未对齐的第二个视图的标签。√

all_data:所有数据的数据集。

[0]表示第一个视图的对齐数据＋未对齐数据，[1]表示第二个视图的对齐数据＋未对齐数据。√

all_label_X:第一个视图的对齐数据+未对齐数据 的标签。√

all_label_Y:第二个视图的对齐数据+未对齐数据 的标签。√

S_cosX:第一个视图的余弦相似度图。√ align*unalign

S_cosY:第二个视图的余弦相似度图。√ align*unalign

S_X：第一个视图的强化锚点图。√ align*unalign

S_Y:第二个视图的强化锚点图。√ align*unalign

unmapach_X:第一个视图的强化锚点图。√ unalign*align

unmapach_Y:第二个视图的强化锚点图。√ unalign*align

viewnum:数据集的视图数量,在这是2。√

label_num:数据每个视图的标签数量。√

cluster_num:聚类的数量。√

align_outX：对齐后的第一个视图的数据。√（强化锚点图作为属于源数据）

align_outY：对齐后的第一个视图的数据。√

--------------------

这四个就是新的对齐的数据和不对齐的数据。

cmapX:第一个视图的余弦距离图。√ align*align

campY:第二个视图的余弦距离图。√ align*align

cunmapX:第一个视图的余弦距离图。√ align*unalign，S_cosX.T。

cunmapY:第二个视图的余距离图。√ align*unalign，S_cosY.T。

---------



numalign:对齐数据集的数量。√

adjacency_matrixX：第一个视图不对齐部分之间的邻接矩阵。之前×，现在√。

adjacency_matrixY：第二个视图不对齐部分之间的邻接矩阵。之前×，现在√。

align_out：包含align_outX和align_outY。√

kn：包含adjacency_matrixX和adjacency_matrixY。√

input_features：输入模型的维度。√

output_features：输出模型的维度。√

con_graph：对比矩阵

condataX：第一个视图加第二个视图已知对齐的部分。

condataY：第二个视图加第一个视图已知对齐的部分。

combined_arrayposX：第一个视图各个样本对应的正对索引。

combined_arrayposY：第二个视图各个样本对应的正对索引。

combined_arraynegX：第一个视图各个样本对应的负对索引。

combined_arraynegY：第二个视图各个样本对应的负对索引。

dataposX：第一个视图各个样本对应的正对。

datanegX：第一个视图各个样本对应的负对。

dataposY：第一个视图各个样本对应的正对。

datanegY：第一个视图各个样本对应的负对。

datapairX：第一个视图各个样本对应的正对和负对连接到一起。

datapairY：第二个视图各个样本对应的正对和负对连接到一起。

------------------------------------------------------------------------

错误：

adjacency_matrixX：第一个视图不对齐部分之间的邻接矩阵。之前×，现在√。

adjacency_matrixY：第二个视图不对齐部分之间的邻接矩阵。之前×，现在√。

之前GCN输入的数据和邻接矩阵不对应。

现在训练的视图数据和对比视图的数据不对应，对比视图没有对齐

-----------------------------------------------------------------------------------------------

模型那出了一些问题需要进一步验证。