FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

RUN pip install "opencv-python-headless<4.3"
RUN pip install -U albumentations --no-binary qudida,albumentations
RUN pip install cython scikit-learn pandas fastai accelerate pytorch-lightning loguru pytorch_pfn_extras
RUN pip install git+https://github.com/rwightman/pytorch-image-models.git
RUN pip install git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer.git

RUN pip install -U ipywidgets 
RUN jupyter nbextension enable --py widgetsnbextension

RUN pip install ml_collections ipyexperiments wandb
RUN wandb login dada17887d4a4c82baeb59905969b7614d892bde

WORKDIR /workspace

RUN pip install -U jupyterlab
RUN conda install -c conda-forge nodejs -y
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
