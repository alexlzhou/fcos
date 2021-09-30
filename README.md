## Fully Convolutional One-Stage Object Detection

This is an implementation of Fully Convolutional One-Stage Object Detectors (https://arxiv.org/abs/1904.01355) using only PyTorch libraries.

FCOS is anchor box free and proposal free. It employs the "centerness" idea to determine which box a pixel belongs to, incase there is an overlap of bounding boxes. FCOS is fast to train, accurate, and can be trained end-to-end.
