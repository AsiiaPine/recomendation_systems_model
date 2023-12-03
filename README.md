# Task explanation:

Create a recomendation system solution for MovieLens dataset.
The dataset is in the [directory](data/raw/).

# Models
Two models are presented in the directory:
One was trained in ipynb file (LightGCN)
LayerGCN was trained with [LayerGCN main](/models/IMRec/main.py).

The result for both models are well-described in [report](/reports/final_report.md):

- LightGCN:
Model weights are stored in [lightgcn.pth](/models/lightgcn.pth).
The [LightGCN notebook](/notebooks/light_gcn.ipynb) solution is based on [Recommender Systems with GNNs in PyG](https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377).
![LightGCN](https://pic2.zhimg.com/v2-c55e97f70743eee133893b5ec5a15d8d_r.jpg)
![img](reports/figures/lightgcn_20_epochs_metrics.png)
![img](reports/figures/lightgcn_20_epochs_loss.png)
- LayerGCN:
Solution is based on [GitHub](https://github.com/enoche/ImRec/tree/master), the repo structure is was too complicated to split the solution into .py and .ipynb files, therefore I modified the repo to create plots and will make pull request. For now, the modified repo version is in the [folder](/models/IMREC/)
There is no well-performed model, therefore all pretrained models are [saved](/models/IMRec/saved/)
![LayerGCN](reports/figures/layergcn.png)

![img](models/IMREC/plots/best_eval_(999,%204,%200.2,%200.1)_0.1225.png)
![img](models/IMREC/plots/best_(999,%204,%200.2,%200.1)_0.1225.png)