<div align="center">

![Logo](assets/raspberry.png)

# Berries 

**Pytorch wrapper for faster prototyping.**

**!This project is under heavy development and not ready to use.**

</div>


## Installation:

 * Download cuda from [nvidia-cuda-11.0](https://developer.nvidia.com/cuda-11.0-download-archive)
 * Clone the **[berries](https://github.com/ugurgudelek/berries)** repository
 * Enter into the project **root** directory
 * `conda env create --file environment.yml`
 * `conda activate berries`
 * `conda develop .`




## Manual installation:

 * Download cuda from [nvidia-cuda-11.0](https://developer.nvidia.com/cuda-11.0-download-archive)
 * `conda create --name berries python=3.8`
 * `conda activate berries`
 * `conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -y`
 * `conda install yapf -y`
 * `conda install pandas -y`
 * `conda install pyyaml -y`
 * `conda install -c conda-forge matplotlib -y`
 * `conda install scikit-learn -y`
 * `conda install -c plotly plotly -y`
 * `conda install -c conda-forge tqdm -y`
 * `conda develop .`




## Package export

```conda env export --from-history > environment.yml```

```conda env export > environment-strict.yml```

then fix `environment.yml` package version and channels using environment-strict.yml
Also create environment-soft.yml to ease versions