# Overlapping H2D copies with Compute using Torch

## N: 1000	
```
sync
totalTimer: 0.14412593841553	

async
totalTimer: 0.14150881767273	
```

## N: 2000	
```
sync
totalTimer: 0.18196105957031	

async
totalTimer: 0.16074895858765	
```

## N: 4000	
```
sync
totalTimer: 0.33683300018311	

async
totalTimer: 0.27457714080811	
```

## N: 8000	
```
sync
totalTimer: 1.4027299880981	

async
totalTimer: 1.2087471485138	
```

## double-buffer.lua

Toy example of how one might convert Soumith's Multi-GPU ImageNet code
to an async version.

The critical component is the `trainBatch` in Soumith's code copies the data
then computes on it.

In this verison, `CopyAndCompute` handles the double buffering, and `Compute`
does the heavy lifting.
