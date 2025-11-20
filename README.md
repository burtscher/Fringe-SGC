## Fringe-SGC

## Description
Fringe-SGC is a CUDA code for counting the number of occurrences of a subgraph in a larger graph. It decomposes the user-defined subgraph into core and fringe vertices, searches the graph for the core vertices, and then utilizes a formula to rapidly count the number of occurrences.<br>

## Installation
1. To begin, ensure that you have a machine with a CUDA-capable GPU with compute capability of at least 8.0 and CUDA toolkit 11.1.<br>
2. Make sure gnuplot is installed to plot the results in bar graphs (we used version 6.0).<br>

3. Follow these instructions to reproduce the results of the paper:<br>
     a) Clone the code using "git clone https://github.com/burtscher/Fringe-SGC.git"<br>
     b) Navigate to the Fringe-SGC directory by executing "cd Fringe-SGC"<br>
     c) Adjust the -arch flag in the Makefile for your GPU architecture<br>
     d) Execute "make"<br>
     e) Run "python3 figures_Fringe.py"<br>
     f) View the resulting 7 PNG files<br> 

4. If you want to create your own subgraphs, follow the above steps with this next step:<br>
     a) ./src/fringePreprocess {number of nodes} {edge1_source} {edge1_destination} ...<br>
     b) ./src/fringeCount {graph input} motif.mo<br>

## Publication
If you use Fringe-SGC in your work, please cite the following publication:
Cameron Bradley, Ghadeer Alabandi, and Martin Burtscher. Fringe-SGC: Counting Subgraphs with Fringe Vertices. Proceedings of the 2025 ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis. November 2025. [[doi]](https://doi.org/10.1145/3712285.3759839) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3712285.3759839) [[slides]](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fuserweb.cs.txstate.edu%2F~burtscher%2Fslides%2Fsc25a.pptx&wdOrigin=BROWSELINK)

*This work has been supported by the National Science Foundation under Award Number 1955367.*


