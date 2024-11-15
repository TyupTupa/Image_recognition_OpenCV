# Image recognition recognition

_using the OPENCV library._

## Description 

This program uses reference pictire to find similar objects on video stream. Uses SIFT (scale-invariant feature transform) algorithm that is a feature detection algorithm in computer vision for identifying and describing local features in images. 

The algorithm identifies key points in the image and compares them with points in the reference picture. This is why the algorithm works well on high-contrast frames.

My university logo is taken as a reference image.

## Installation

Install project using:

```bash
git clone https://github.com/TyupTupa/Image_recognition_OpenCV
```
Create Python virtual enviroment:

```bash
python -m venv venv
```
Activate:

```bash
venv\Scripts\activate
```

Run to find the desired requirements file and install it in your current environment:

```bash
pip install -r requirements.txt
```
