## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[vehicle_and_non_vehicle]: ./examples/vehicle_and_non_vehicle.png
[hog_example]: ./examples/hog_example.png
[final_hog]: ./examples/final_hog.png
[sliding_windows]: ./examples/sliding_windows.png
[test_image_pipeline_1]: ./examples/test_image_pipeline_1.jpg
[test_image_pipeline_2]: ./examples/test_image_pipeline_2.jpg
[test_image_pipeline_3]: ./examples/test_image_pipeline_3.jpg
[heatmap]: ./examples/heatmap.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells in the "Features extraction" section of the Jupyter notebook "./Vehicle-Detection.ipynb".

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][vehicle_and_non_vehicle]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed one random image from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(14, 14)` and `cells_per_block=(2, 2)`:

![alt text][hog_example]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of HOG parameters and color spaces and settled on the following:

* orientations: 10
* pixels_per_cell: (14, 14)
* cells_per_block: (2, 2)

![alt text][final_hog]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the code cells in the "Classification" section of the Jupyter notebook "./Vehicle-Detection.ipynb".

I used HOG features, binned color features and color histogram features to train my classifier.

Once I had extracted these features from all the images, I normalised them.

Then I splitted my training and test data.

I chose to use a SVM and carried out an automatic parameter search with `GridSearchCV`. Unfortunately this took too long
so I had to cancel it and asked fellow students what they had chosen as hyper-parameters for their SVM. I combined that
knowledge with the parameters used in the course and I ended up with this:

* kernel: `rbf`
* C: `100`
* gamma: `auto`

The final step was to train it against my training data.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the code cells in the "Sliding Window Search" section of the Jupyter notebook "./Vehicle-Detection.ipynb".

I used a the HOG sub-sampling window search with scale factors of 1, 1.3, 1.5 and 1.8. I restricted the area of the image to perfom the search on based on the scale factor
(no need to have large detection windows for in the middle of the image, large objects will be at the front).

With this technique we extract the HOG features once for each of my scaling factors and then we sub-sample the features to get all the overlaying windows.

Here's an example of sliding windows search with successful detections:

![alt text][sliding_windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.

Here are some example images of each stage of the pipeline:

![alt text][test_image_pipeline_1]

![alt text][test_image_pipeline_2]

![alt text][test_image_pipeline_3]

I tried improving the running performance by using the OpenCV HOG function but I ran into a lot of issues with sub-sampling and it
caused a lot of bad predictions.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_videos/project_video_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the code cells in the "Pipeline" section of the Jupyter notebook "./Vehicle-Detection.ipynb".

I keep a list of bounding boxes detected by my HOG sub-sampling window search over the last 6 frames. I create a heatmap of these bounding boxes and then threshold that map to identify vehicle detections.
This allow me to surface only the bounding boxes that I'm sure of.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap and create bounding boxes to cover the area of each blob.

Here's an example of an image with bounding boxes created based on a heatmap:

![alt text][heatmap]

Finally, I keep track of detected bounding boxes over time and only draw the bounding boxes of objects that have been detected over more than 2 frames. This helps reducing the number of false positives.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

* Because of the way I'm combining bounding boxes from heatmaps if two vehicles are close I get a single bounding box instead of two. This is bad because the self-driving car
might take different actions if it believe there is only vehicle instead of two. There must be a way to differentiate the vehicles even if the bounding boxes are super close.

* Due to the parameters of my sliding window search I'm missing some detections on the right edge of the videos. This means I don't detects cars as early as possible.

* The pipeline is really slow. It wouldn't work in a real-time scenario. Given more time I'd like to implement an architecture based on [YOLO](https://pjreddie.com/darknet/yolo/).

