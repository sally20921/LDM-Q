## References 
- [Distributed Deep Learning with PyTorch Lightning](https://devblog.pytorchlightning.ai/distributed-deep-learning-with-pytorch-lightning-part-1-8df1d032e6d3)


## PyTorch Lightning for distributed multi-GPU training

PyTorch Lightning makes your PyTorch code hardware agnostic and easy to sacale.

You can run on a single GPU, multiple GPUs, or even multiple GPU nodes (servers) with zero code change. 

The next level to distributed multi-GPU training is multi-node training.
Multi-node refers to scaling your model training to multiple GPU machines on-premise and on the cloud. 


## Distributed Multi-GPU training

- enables speed up training when you have a large amount of data 
- work with large batch sizes that cannot fit into the memory of a single GPU
- have a large model parameter count that does not fit into the memory of a single GPU

DDP, or Distributed Data-Parallel approach splits data evenly across the devices. 
It is the most common use of multi-GPU and multi-node training today.
The first two cases can be addressed effectively by DDP.

In the data-parallel distributed setting, the data is split evenly across all GPUs. Each GPU will train with its own data and synchronize the gradients with all others.


The third case where the model is so large that it does not fit into the memory is rather exotic 
but has seen very impressive developments recently from `DeepSpeed` and `FairScale`.




## `trainer.x` syntax from LightningCLI 

```bash
python train.py --trainer.accelerator ddp --trainer.gpus 8 
```

### multi-gpu setting

```bash

trainer = Trainer(gpus=8, accelerator="ddp")
```

### multi-node setting

```bash

trainer = Trainer(gpus=8, num_nodes=4, accelerator="ddp")
```





















