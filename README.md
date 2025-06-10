# PC-TED: A Learning-based Point Cloud Compression Toolkit For Training, Evaluation and Deployment

## Introduction

![](./assets/fig_overview.jpg)

PC-TED is a unified toolkit bridging research and practical application for learning-based point cloud compression. By consolidating training, evaluation, and deployment workflows, PC-TED facilitates systematic comparisons of compression algorithms with an emphasis on real-world applicability.

## Software Architecture

- **Model Zoo:** we have included common point cloud compression methods in the model zoo.
- **Compression Evaluator:** we currently include some metrics for evaluating the compression performance of algorithms.
- **Inference Evaluator:** we aim to provide hardware platform-independent metrics for evaluating inference performance.
- **Model Exportor:** the model exporter is to convert networks into the Open Neural Network Exchange (ONNX) format.

## Installation
```bash
git clone —recursive https://gitee.com/rolfma/pc-ted.git
cd pc-ted
chmod +x install.sh
./install.sh
```

## Usage
- Model Zoo: Please refer to `Examples/model_zoo.md`.
- Compression Evaluator and Inference Evaluator: Please refer to `Examples/evaluator.py`.
- Model Exportor: Please refer to `Examples/exportor.py`.

## Algorithm Comparasion
### Compression performance measured by bits per point (bpp) of some lossless intra-frame methods in the model zoo.
![](./assets/lossless.jpg)
### Compression performance presented by rate-distortion curves of some lossy intra-frame methods in our model zoo.
![](./assets/lossy.jpg)




