# SAM Image Segmentation Tool
This script provides a simple GUI tool for image segmentation using the SAM model locally. The tool allows the user to load an image, select a prompt mode (single point or bounding box), and then select a point or draw a bounding box on the image to generate a segmentation mask. The mask or cutout can be saved as an image file. The tool uses the SAM model to generate the segmentation mask based on the user input.

![image SAM gui tool example](https://raw.githubusercontent.com/B-Bracke/SAM-Segmentation-Tool/main/SAM%20gui%20tool.jpg)


## Installation
To install the dependencies, you can use the provided environment.yml file to create a conda environment.

1. Clone the repository:
```sh
git clone <repository_url>
cd <repository_directory>
```

2. Create the conda environment:
```sh
conda env create -f environment.yml
```

3. Activate the environment:
```sh
conda activate SAM
```

## Usage
To execute the script, run the following command in the terminal:
```sh
conda activate sam-gui-tool
```

## How to Use the GUI
Open the Tool: Run the script to open the GUI.
Load an Image: Click the "Open Image" button to load an image from your file system.
Select Prompt Mode: Choose between "single point" or "bounding box" mode.
Generate Segmentation Mask:
For single point mode, click on the image to select a point.
For bounding box mode, click and drag to draw a bounding box on the image.
Save the Mask: Click the "Save Mask" button to save the generated segmentation mask as an image file.
Save the Cutout: Click the "Save Cutout" button to save the generated segmentation cutout as an image file.
