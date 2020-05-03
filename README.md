# Gaussian Mixture Models (GMMs)
To see a small graphical example of why we use GMMs, see: https://www.desmos.com/calculator/i8ykgpmgxa

## Simple GMM Visualization Tool
Using the visualization tool, you can quickly get a feel for how GMMs are trained, at least in 2D. 
All the required packages can be found in ```requirements.txt```.

### Fitting a GMM to a simple distribution
Using the following line:
```bash
python visualize_gmm.py -k 5 -m 5
```
Produces a simple demo with data points that were generated from 5 random Gaussians (controlled by the argument ```-m```), that is fitted to a GMM with 5 clusters (controlled by the argument ```k```), similar to the following:
![Demo GMM](https://github.com/friedroy/gaussians/blob/master/examples/demo.gif)

### Fitting a GMM to a more complex distribution
Any 2D data, saved as numpy ndarray in a ```.npy``` file, with shape \[N, 2\] can also be fitted to a GMM, using this tool; the following line demonstrates how:
```bash
python visualize_gmm.py -k <choose number of clusters> --load_path <your .npy file path here> [--print_ll]
```
You can add the flag ```--print_ll``` to track progress. An example of how to load data:
```bash
python visualize_gmm.py -k 20 --load_path examples\circles.npy --print_ll -i 100 --fps 15
```
The result should be something similar to:
![Demo Circles](https://github.com/friedroy/gaussians/blob/master/examples/circles.gif)
