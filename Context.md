# Objective 

Predict the speed at which a pet is adopted based on the pet's listing data.

# Background

PetFinder is an animal welfare platform that connects stray animals to families. For more information about what they do or if you want to adopt a pet, check [here](https://www.petfinder.my/).

# Data

Quoted from [here](https://www.kaggle.com/c/petfinder-adoption-prediction/data).

**Note** A profile can represent a group of pets.

## Files

  * train.csv - Tabular/text data for the training set
  * test.csv - Tabular/text data for the test set
  * sample_submission.csv - A sample submission file in the correct format
  * breed_labels.csv - Contains Type, and BreedName for each BreedID. Type 1 is dog, 2 is cat.
  * color_labels.csv - Contains ColorName for each ColorID
  * state_labels.csv - Contains StateName for each StateID

## Data Fields

  * PetID - Unique hash ID of pet profile
  * AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
  * Type - Type of animal (1 = Dog, 2 = Cat)
  * Name - Name of pet (Empty if not named)
  * Age - Age of pet when listed, in months
  * Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
  * Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
  * Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
  * Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
  * Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
  * Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
  * MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
  * FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
  * Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
  * Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
  * Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
  * Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
  * Quantity - Number of pets represented in profile
  * Fee - Adoption fee (0 = Free)
  * State - State location in Malaysia (Refer to StateLabels dictionary)
  * RescuerID - Unique hash ID of rescuer
  * VideoAmt - Total uploaded videos for this pet
  * PhotoAmt - Total uploaded photos for this pet
  * Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.

For pets that have images: 
  * named in the format of PetID-ImageNumber.jpg

PetFinder ran the images through Google Vision's API and the text through Google's NLP API.

## Labels

  * 0 - Pet was adopted on the same day as it was listed. 
  * 1 - Pet was adopted between 1 and 7 days (1st week) after being listed. 
  * 2 - Pet was adopted between 8 and 30 days (1st month) after being listed. 
  * 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed. 
  * 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).

# Acknowledgments

This project idea is from [Kaggle](https://www.kaggle.com/c/petfinder-adoption-prediction).

Thanks to [PetFinder](https://www.petfinder.my/) for providing the data!
