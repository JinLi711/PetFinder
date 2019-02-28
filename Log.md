This is the log of all the progress made on the PetFinder project. Hope everything goes well!

# January 20, 2019

**Today's Progress**

1. Started up by downloading the data and creating starter files.

2. Created steps in my [Plan.md](https://github.com/JinLi711/PetFinder/blob/master/Plan.md) Completed some objectives (as described in Plan.md). Tried to understand the features and the categories (or numbers) in each feature.

**Thoughts** Although the images and text have been ran through Google Cloud's API, I think it would be interesting to build my own neural networks to do this analysis (since Google Cloud's API is for a general analysis)

# January 25, 2019

Decided its probably more efficient to describe my thoughts in this log, since I describe what I accomplished in Plan.md.

**Thoughts** Definitely going to do my own analysis on pictures and text since GCP results aren't that informative.

# January 30, 2019

**What I Could Have Done Better** 

  * Instead of visualizing through Python notebooks, visualize it through a framework like flask.
  * How do I unittest visualizations?
  * Create classes even before creating visualizations.
  * Are there better places to put my observations other than in the python notebooks?

# January 31, 2019

**What I Could Have Done Better** 

  * Get in the better habit in not using spaces.
  * Get in the habit of using " " for strings and ' ' for characters.

**Questions**
  * How do I get images without loading everything at once? Turns out to be very tricky. Even trickier to deal with is the fact that one instance can have multiple images.

# February 1, 2019

**How To Deal With Images**
  * there can be multiple images for each instance. The number varies. Things I can do:
  * duplicate each instance with multiple images. The only difference between the duplicates is the attached image.
    * The problem with this is that the images become independent, which is a problem because when a person looks, they look and decide with all images at once.
    * To generalize to the validation set, I guess I can split it, predict for each image, then average the result. 
    * Probably will do this because it captures the most information.
  * randomize the image everytime it is resampled when training.
  * only pick the first image.
  * If no image, just do entirely black or white? Probably just black (since RBG representation is all 0)

**How To Combine All the Types of Data**
  * Trickiest part: combining three different types of data to the same model. Keras kinda makes it hard to do so.
  * How I'm going to do this:
  * Import raw tabular data and preprocessed tabular data. The old tabular data is only used for extracting the descriptions (preprocessed tabular data does not contain descriptions).
  * I would need to write my own image generator [Here for some help](https://github.com/keras-team/keras/issues/3386).
  * Split the jpg files column from the other tabular data. 