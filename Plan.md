C - Complete
D - Doing
NW - Not working on yet
ND - Decided not to do this


# Step 1: Set Up

| Progress | Date Finished | Task                  
|----------|---------------|-----
| C        | 1/20/2019     | Download The data    
| C        | 1/20/2019     | Create Plan.md, Context.md, Log.md, REAME.md
| C        | 1/20/2019     | Open the Zip Files
| C        | 1/20/2019     | Add license   

# Step 2: Visualization

| Progress | Date Finished | Task                  
|----------|---------------|-----
| C        | 1/20/2019     | Look into the data (look into the directory structures and whatnot)
| C        | 1/20/2019     | Look into the csv files
| C        | 1/21/2019     | Create histograms for labels and other categorical features.
| NW | NAN | Visually compare whether images match the descriptions.
| C        | 1/21/2019     | Create wordclouds for text (and pet names).
| C        | 1/21/2019     | Create bar graphs to show relations between features and labels.
| C        | 1/25/2019     | Visualize some images from the train set.
| C        | 1/25/2019     | Visualize some results from GCP analysis (sentiment on descriptions and metadata from images)
| ND | NAN | Create heat maps for correlations
| C        | 1/30/2019     | Visualize the descriptions.

# Step 3: Preprocess

| Progress | Date Finished | Task                  
|----------|---------------|-----
| C        | 1/26/2019     | Compress data to reduce memory.
| C        | 1/27/2019     | Get data and split into train and validate set.
| C        | 1/27/2019     | Split the label from the train data.
| C        | 1/30/2019     | Import Stanford's GloVe.
| C        | 1/30/2019     | Convert the words in the descriptions to vector representations.
| ND       | NAN | Get the meta data from GCP results.
| ND       | NAN | Get images if an image is avaliable.
| ND       | NAN           | Label whether image exists.
| C        | 1/31/2019     | Deal with missing values
| C        | 1/31/2019     | Deal with names category.
| C        | 1/31/2019     | Deal with values that do not make sense.
| C        | 1/31/2019     | Deal with outliers for numeric data.
| C        | 1/31/2019     | Standard scaler on numerics
| C        | 1/31/2019     | One hot encoding for categories.
| C        | 1/31/2019     | Get the test data.
| C        | 2/1/2019      | Duplicate rows so each row links to a different image.


# Step 4: Model Building

| Progress | Date Finished | Task                  
|----------|---------------|-----
| NW | NAN | Import everything from the preprocessed part.
| NW | NAN | Build a variation of a image segmentation algorithm.
| NW | NAN | Build a RNN for text description to extract relevant information.


# Step 5: Model Evaluation


# Step 6: Final Adjustments 

| Progress | Date Finished | Task  
|----------|---------------|-----
| NW | NAN | Create requirements.txt
| NW | NAN | Create directory tree.
| NW | NAN | Remove warnings.
| NW | NAN | Improve documentations.