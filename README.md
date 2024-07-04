### opr

##### VISUALIZE GRAPH
- We use graphviz to visualize the computation graph
- Currently graph in graph visualization is not available
###### Prequisites
- Graphviz need to be installed
```
    sudo apt-get install graphviz
```
###### Guideline
- Generate a dot file: using the `gen_dot` api of the graph class
```
    graph->gen_dot("output/graph.dot");
```
- Generate graph visualzation image from the dot file
```
    dot -Tpng output/graph.dot -o output/graph.png
```
