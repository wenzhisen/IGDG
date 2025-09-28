from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

name = "tgbl-wiki"

dataset = PyGLinkPropPredDataset(name=name, root="datasets")

dataset.src #all source nodes of edges

