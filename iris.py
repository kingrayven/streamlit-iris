import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd  # Import Pandas for data manipulation

# Load the Iris dataset
def load_data():
    """
    Load the Iris dataset and prepare it for training.
    Returns:
        X (numpy array): Feature matrix
        y (numpy array): Target labels (numerical values: 0, 1, 2)
        df (DataFrame): Full dataset as a Pandas DataFrame
    """
    iris = load_iris()
    X = iris.data  # Features: Sepal Length, Sepal Width, Petal Length, Petal Width
    y = iris.target  # Target: Species (0: Setosa, 1: Versicolor, 2: Virginica)
    
    # Convert the dataset to a Pandas DataFrame
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = y  # Add the target column
    
    return X, y, df

# Train the Linear Regression model
def train_model(X, y):
    """
    Train a Linear Regression model on the Iris dataset.
    Args:
        X (numpy array): Feature matrix
        y (numpy array): Target labels
    Returns:
        model (LinearRegression): Trained Linear Regression model
    """
    model = LinearRegression()
    model.fit(X, y)  # Fit the model to the data
    return model

# Predict the target class
def predict_species(model, sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict the species of the Iris flower using the trained model.
    Args:
        model (LinearRegression): Trained Linear Regression model
        sepal_length (float): Sepal Length
        sepal_width (float): Sepal Width
        petal_length (float): Petal Length
        petal_width (float): Petal Width
    Returns:
        str: Predicted species name ('Setosa', 'Versicolor', 'Virginica')
    """
    # Prepare the input data as a 2D array (required by scikit-learn)
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict the numerical class (0, 1, 2)
    predicted_class = model.predict(input_data)
    
    # Map the numerical class to the species name
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species_map.get(round(predicted_class[0]), 'Unknown')
    return predicted_species

# Main function to run the Streamlit app
def main():
    """
    Main function to create and run the Streamlit app.
    """
    # Set the title of the app
    st.markdown("<h1 style='text-align: center; color: yellow;'>Iris Species Prediction</h1>", unsafe_allow_html=True)
    # st.title("Iris Species Prediction")

    st.markdown("<h2 style='text-align: center;'>Enter the measurements below to predict the species of the Iris flower.</h2>", unsafe_allow_html=True)
    # st.write("Enter the measurements below to predict the species of the Iris flower.")

    try:
        # Load the dataset and train the model
        X, y, df = load_data()  # Load data and get the Pandas DataFrame
        
        # Train the model
        model = train_model(X, y)

        # Create input fields for user data
        # st.subheader("Enter Measurements")
        col1, col2 = st.columns(2)  # Use two columns for better layout
        with col1:
            sepal_length = st.text_input("Sepal Length (cm)", value="5.1")
            sepal_width = st.text_input("Sepal Width (cm)", value="3.5")
        with col2:
            petal_length = st.text_input("Petal Length (cm)", value="1.4")
            petal_width = st.text_input("Petal Width (cm)", value="0.2")

        # Placeholder for displaying the prediction result in the center
        result_placeholder = st.empty()

        # Validate user input
        if st.button("Predict"):
            try:
                # Convert inputs to float
                sepal_length = float(sepal_length)
                sepal_width = float(sepal_width)
                petal_length = float(petal_length)
                petal_width = float(petal_width)

                # Predict the species
                predicted_species = predict_species(model, sepal_length, sepal_width, petal_length, petal_width)

                # Display the result in the center of the screen
                result_placeholder.markdown(
                    f"<h3 style='text-align: center; color: green;'>Predicted Species: {predicted_species}</h3>",
                    unsafe_allow_html=True
                )

                # Show a toast notification
                st.toast(f"Prediction Complete: {predicted_species}", icon="🌱")

            except ValueError:
                st.error("Please enter valid numerical values for all inputs.")
        
        # Display the dataset at the bottom
        st.subheader("Iris Dataset")
        # st.markdown("<h1 style='text-align: center; color: yellow;'>Iris Dataset</h1>", unsafe_allow_html=True)
        st.dataframe(df)  # Show the dataset in a table format

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()