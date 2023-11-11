# Based distance graph embeddings

## What it do?
Model that do the following: for given graph find node embeddings that satisfy following conditions:

- For every embeddings nodes that between have the edge, Euclidian distance less than or equal to one.
- Similar to previous condition, distance for nodes that doesn't have edge should be opposite: greater one.

## What the result?
In a result, you get graph representation in O(n^2) time and graph representation consume O(n log n) memory

O(n log n) because the embeddings dimensionality is O(log n).

## For what it can be used?

For example, you can see in notebook that it can be used to visualize graph in human-friendly way.

I didn't test it but possibly it can be used in Graph neural networks.

## How it works?

Gradient descent optimize the loss function(you can see in DistanceLossLayer class) that is:

```
loss += (1 - c) * self.pushing_func(x[i][j]) + c * self.pulling_func(x[i][j])
```
- x[i][j] is distance between i and j nodes
- c equal to 1 when edge (i, j) exists else 0
- pushing function is max(2 - 2 * x, 0)
- pulling function is max(0.5 * (x - 1), 0)
