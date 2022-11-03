# Dynamic Urban Radiance Fields
This repository contains the Code release for my thesis, 
in which we use Neural Radiance Fields to perform Novel View Synthesis on unbounded dynamic urban scenes. 
We demonstrate our method on the [Waymo Open Dataset](https://waymo.com/open/) 
and data we generate ourselves with [CARLA](https://carla.org).
We combine approaches from [Urban Radiance Fields](https://urban-radiance-fields.github.io) (URF), 
[Mip-NeRF360](https://jonbarron.info/mipnerf360/), [Neural Scene Graphs](https://light.princeton.edu/publication/neural-scene-graphs/) (NSG)
and [BARF](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/).  


Our method is built on top of [Mip-NeRF](https://github.com/google/mipnerf) in [JAX](https://github.com/google/jax)
and in particular contains our own re-implementation of URFs LIDAR losses, 
Mip-NeRF360s spatial re-parameterisation and BARFs frequency encoding filtering.


## Abstract

Modern self-driving systems collect vast amounts of data to train their advanced
detection and decision-making algorithms. In contrast, simulation environments in
which changes to a self-driving system are tested still rely on handcrafted assets and
environment models in many cases. We propose a method that can use the data that is
collected during normal operation of a self-driving car to model a scene in a Neural
Radiance Field (NeRF), which allows us to perform Novel View Synthesis and view the
scene from previously unseen camera perspectives.

To do this we have to overcome a few problems with modeling an unbounded
outdoor scene in a NeRF, namely sampling efficiency in unbounded spaces, sparse
views and dynamic scene content. We do this by combining advances from multiple
recent state-of-the-art works. First we employ a non-linear scene re-parameterization
to deal with sampling efficiency, which shrinks space so it is bounded again and can
be easily sampled. Secondly we deal with sparse views by supervising the densities
that are emitted by a NeRF with depth and volume carving losses. And lastly we
decompose the scene into static background and dynamic parts with the help of 3D
bounding box annotations and then train a separate NeRF for the background and each
dynamic object, which allows us to manipulate camera and object pose separately.

Since we want to make this method easily scalable we relax the requirement for
perfect human-annotated 3D bounding boxes and propose a method to optimise their
position and orientation jointly with the radiance field. We accomplish this by framing
the problem as a camera registration task and treating the transformation given by a
3D bounding box as an extrinsic camera matrix that needs to be registered with image
data. Our method disables positional encoding to get around noisy gradients that can
arise from backpropagating through it until the bounding boxes have converged and
then uses anti-aliased integrated positional encoding to learn high frequency features.

We apply our method to a synthetic dataset which we generate with CARLA, a
self-driving simulator, and the Waymo Open Dataset. Our method is capable of
recovering reasonable 3D bounding boxes from errors up to half a meter from just five
observations when we use LIDAR depth information in addition to RGB supervision.
We also outperform a non-dynamic aware baseline in the Novel View Synthesis task
for both ground truth and optimised bounding boxes.

[Full Thesis](https://drive.google.com/file/d/1e1a-DDODgXoN0TxONETH3ZJ7QWU91tjj/view?usp=sharing)

## Videos

Novel View Synthesis results from our method with rendered depth:
<img src="videos/waymo1_5_combined.gif" width="960">

The results above are generated from a NeRF that was only trained on 25 Images and their corresponding depth images.

In another experiment we jointly optimise the bounding box poses and the radiance field:
<img src="videos/waymo_opt_combined.gif" width="960">
The first gif shows our method when optimising the bounding box pose and the second without optimisation. 

## Installation

Clone this repo. 

```bash
git clone https://github.com/FelTris/durf.git
cd durf
```

Create a new conda environment and depending on what you want to do install dependencies.

```bash
conda create -n durf python=3.6.13; conda activate durf
conda install pip; pip install --upgrade pip
# For waymo data pre-processing use requirements_wod, for carla use requirements_carla
pip install -r requirements_jax.txt
```

Install GPU support for Jax. If you run into trouble here, check for the official Jax installation guide.

```bash
# Remember to change cuda101 to your CUDA version, e.g. cuda110 for CUDA 11.0.
pip install --upgrade jax jaxlib==0.1.65+cuda101 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Data

For the waymo open data, download the sequences from the official website and run both the waymo_data and waymo_labels notebooks.
To generate the sky masks use an off-the-shelf segmentation network and save the sky pixels as a mask.
We also provide a few examples 
[here](https://drive.google.com/drive/folders/1FENBETwX2K_8qdYIckfUGiLUtrmyol1T?usp=sharing) for data we generate with CARLA.
If you want an example of how the waymo data format is supposed to look, please contact me directly. 
I'm not sure how ok it is otherwise to share waymo open data publically. 

## Training \& Inference

We provide example scripts for training with CARLA and Waymo data in `scripts/`. 
Change the paths to where you put the data on your system. 
[Gin](https://github.com/google/gin-config) configuration files are provided for each dataset in `configs/`. 
To evaluate or render your own trajectory we provide jupyter notebooks in `notebooks/`.

## Citation

If you use this code in your research please cite our work.

```
@mastersthesis{durf22,
  author  = {Felix Tristram and
               Matthias Niessner},
  title   = {Neural Rendering for Dynamic Urban Scenes},
  school  = {Technical University of Munich},
  year    = {2022},
  note    = {Visual Computing Group}
}
```