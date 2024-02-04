# A-Multi-Modal-Approach-to-Improve-Scene-Context

## WHAT AND WHY?
Scene classification is a crucial tool in computer vision, helping organize and categorize the massive influx of daily-uploaded internet images. With diverse applications such as object detection and activity recognition, it impacts areas like blind assistance, autonomous vehicles, robotics, and augmented reality. Our study proposes an integrated approach using CNNs and LSTM for efficient scene evaluation, demonstrating a solid 81.3% test accuracy with our custom model. This highlights the importance of accurate and efficient scene classification algorithms in computer vision research and development.

## OVERVIEW
Now notice the image below. 
<p align="center">
  <img src="https://github.com/jayant1211/A-Multi-Modal-Approach-to-Improve-Scene-Context/blob/main/images/image.png" alt="Image" width="50%" height="50%">
</p>
A certain scene classification model predicts this as: 
<br>1. religious_procession_scene - 90% confidence
<br>2. Farm - 7% confidence
<br>3. Hospital - 3% confidence 
(wrongly, the correct should be farm)
<br>
<br>
and certain caption_generator generates this captions:
<br>1. Two Ladies walking in green field - 99% confidence
<br>2. Two people in bus - 0.5%
<br>3. People in vehilce - 0.5%

We can see the caption with highest confidence is correct, but scene_prediction with highest confidence is not.
<br>
<br>
We employ an integrated filtering approach, we utilizes this confidences to make out the best suitable pair among caption and scene_category.
<br><br>
So based on our filtering approach, for that wrongly classified image, we would get result as something like:
"Scene is of the {farm} and there are {} two people are sitting in front of store with greenery all in background"
<br>
## APPROACH

### SCENE CLASSIFICATION MODEL
We have employed 2 models to classify scene among certain categories.

#### A
first, Places365 CNN model - This model is pre-trained on Places365 Dataset which is a state- of-the-art scene-centric dataset that contains 365 categories. The authors of Places365 have developed and released convolutional neural networks (CNNs) with a range of different architectures trained on the Places365 dataset. The model we used is InceptionV3. While not the most accurate among all the available pre-trained CNNs released by the authors, it does provide good trade-off between computation efficiency and performance.
<br><br>

#### B
secondly, we have created Custom Scene Classification Model for Indian Environments. We have merged some of the popular categories among places365 and added some of our own(please refer paper for details). Our approach is designed to focus on smaller but more frequently occurring categories that are specific to Indian environments. We have used InceptionV3 architecture for training. 

<p align="center">
  <img src="https://github.com/jayant1211/A-Multi-Modal-Approach-to-Improve-Scene-Context/blob/main/images/model-block.png" alt="Image" width="50%" height="50%">
</p>

### IMAGE CAPTION GENERATOR
We have trained a CNN-RNN model for the image caption generator using the state-of-the-art flickr8k dataset. The choice of CNN in this study is the Xception network, which is pre-trained on ImageNet for 1000 categories, due to its exceptional feature extraction capabilities.

Traditional image caption generators typically produce a single caption, in proposed study, a heuristic approach, beam search algorithm is being used for generating multiple captions. For each caption a score corresponding to it is generated. Score generated for each caption represents the model's level of confidence in its generated output. Integrated filtering mechanism utilizes these confidence scores to enhance the model's understanding of the scene being depicted in the image.

<br>For detailed description, please refer the paper.

<br>Output:
<p align="center">
  <img src="https://github.com/jayant1211/A-Multi-Modal-Approach-to-Improve-Scene-Context/blob/main/images/captions.png" alt="Image" width="50%" height="50%">
</p>

### INTERGRATED APPROACH FOR IMPROVED RESULTS
The idea of the Classification CNN and image captioning module to validate each other’s results can generate more correct outputs. For validating results, a person can tell the relevance between given places and captions. We have integrated exsiting LLM APIs, to tell the relevance between places and captions. Based on the result, an improved output is given.
<br>
Based on multiple scene options and caption options we have, we select one as base module and one as to be filtered. The base selection is purely done on confidence scores. Once we have the base, we generate a prompt something as:

<br>

```bash
Only one answer. Print the corresponding number of correct option: which of the following descriptions match can be most suitable for {} place: - {}
```

<br>and this prompt is fed to GPT API. And in response we get a most suitable pair of (scene, caption)


### RESULTS
#### 1. Custom model - Scene Classification
<p align="center">
  <img src="https://github.com/jayant1211/A-Multi-Modal-Approach-to-Improve-Scene-Context/blob/main/images/confusion-matrix.png" alt="Image" width="50%" height="50%">
</p>

| Metric                | Accuracy  |
|-----------------------|----------:|
| Train Accuracy        |   90.32%  |
| Validation Accuracy   |   80.3%   |
| Test Accuracy         |   81.2%   |


#### 2. Caption Generator
The average BLEU score for up to 4 grams for the image caption generator tested for 1000 different images is given in the table below:
| Metric                |           |
|-----------------------|----------:|
| B-1                   |   0.25    |
| B-2                   |   0.10    |
| B-3                   |   0.05    |
| B-4                   |   0.03    |

#### 3. Integrated approach result samples
<p align="center">
  <img src="https://github.com/jayant1211/A-Multi-Modal-Approach-to-Improve-Scene-Context/blob/main/images/improved_result_samples.png" alt="Image" width="75%" height="75%">
</p>
