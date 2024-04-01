# Titanic Survival Prediction

This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques.

## Description

The code provided in this repository analyzes the Titanic dataset, which includes information about passengers such as their age, gender, ticket class, cabin, and port of embarkation. The goal is to build machine learning models that predict whether a passenger survived the Titanic disaster.

## Installation

To run the code, make sure you have Python installed on your system. Additionally, you'll need the following Python libraries:

- pandas
- numpy
- seaborn
- scikit-learn
- matplotlib
- missingno

You can install these dependencies using pip:
!pip install pandas 
!pip install numpy 
!pip install seaborn 
!pip install scikit-learn 
!pip install matplotlib 
!pip install missingno

## Usage

1. Clone this repository to your local machine.
2. Ensure you have the required dataset files (`train.csv` and `test.csv`) in the same directory as the provided code.
3. Open a terminal or command prompt and navigate to the directory containing the code.
4. Run the Python script using the following command:


5. The script will process the data, train machine learning models, and generate predictions for the test dataset.

## File Structure

- `titanic_survival_prediction.py`: Python script containing the code for data preprocessing, model training, and prediction.
- `train.csv`: Dataset containing training data.
- `test.csv`: Dataset containing test data.

## Examples

### Example 1: Data Preprocessing

The code preprocesses the Titanic dataset by handling missing values, encoding categorical variables, and preparing the data for model training.

```python
# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the Titanic dataset
train_data = pd.read_csv('train.csv')

# Handling missing values
imputer = SimpleImputer(strategy='mean')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])

# Encoding categorical variables
encoder = OneHotEncoder()
encoded_variables = encoder.fit_transform(train_data[['Sex', 'Embarked']])

# Print the preprocessed data
print(train_data.head())
```

Example 2: Model Training and Prediction

```python
# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data.drop('Survived', axis=1), train_data['Survived'], test_size=0.2, random_state=42)

# Initializing and training the Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Making predictions on the test set
predictions = rf_classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
```

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

## License

This project is licensed under the MIT License.

## Credits

This project was created by Minata Coulibaly. Special thanks to the Kaggle Titanic dataset for providing the data.

## Contact

For questions or feedback, please contact minatamarie@gmail.com.
