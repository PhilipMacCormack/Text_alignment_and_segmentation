# Text_alignment_and_segmentation
This project was part of my Master's Thesis Project during spring 2023

The system takes an image of a handwritten page document as input and segments and aligns the image to a ground truth. In the case that a ground truth is not available, the algorithm allows for manual transcription of the segmentation. Where the segmentation fails to recognize text, it is possible to correct the boxes during the process. Bayesian optimisation is used for automatically setting reasonable parameters.
The resulting report from the thesis project can be found at:

https://www.overleaf.com/project/63c523e1bc389181760c12e3

## Prerequisites
To be able to run the pipeline in its entirety, see ```requirements.txt``` for required packages.

## Usage
* Clone this repository by using:
```
> git clone https://github.com/PhilipMacCormack/Text_alignment_and_segmentation
```
* cd into ```Text_alignment_and_segmentation```:
* Install packages from ```requirements.txt```
* Open ```main.py```
* Input parameters in the script
* Run ```main.py```
```
> python main.py
```
* Follow the procedure from the terminal until finish

## Output
The output from the algorithm can be found in ```Text_alignment_and_segmentation/Results/{file}```. The output consists of several saved images from the process as well as an xml file containing the final alignment of the document. Individual line, and word images from the segmentation can also be found, in ```results/{file}/lines```.


## References
This algorithm is partly based upon work from:

https://github.com/KadenMc/PreprocessingHTR

https://github.com/harshavkumar/word_segmentation
