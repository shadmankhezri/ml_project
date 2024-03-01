"""
First, we read the data, and then using the description() and show_info_data() functions, we obtained
general information about the dataset, and for example, we found that the data does not contain Null 
information. And since we want to use simple linear regression in this example, we only need one of the 
features in the dataset to create a sample, and we have to check to see which of the features of the data 
is more suitable for building the model. We can look at the graph of each of the car's features based on 
the amount of  co2emission. Finally, in this example, we came to the conclusion that the most suitable option 
for making the model is the engine size.


"""

# Importing functions from the modules inside the src

from src.display import disply
from src.read_file import read_file
from src.description import description , show_info_data
from src.count_engine_size import show_histogram
from src.scatter_engine_emission import show_scatter_engine_emissions
from src.split_train_test import split_train_test
from src.train_linear_model import train_linear_model
from src.show_train_test_plot import show_train_test_plot
from src.error_score import calculate_metrics

#---------------------------------------

PATH = "data/FuelConsumption.csv"

#---------------------------------------

def main():
    
    # make our data frame

    df_fuel = read_file(PATH)
    disply(f"\nShow the first data frame of Fuel Consumption:\n\n{df_fuel}")

#---------------------------------------

    # General display of information about the data frame

    df_description = description(df_fuel)
    disply(f"\nShow some information about data frame:\n\n{df_description}")


    show_info_data(df_fuel)
    print(100*"*")

#---------------------------------------

    """
    We can separate the features that may have the greatest impact on the co2emissions from the 
    dataframe and check these features one by one to finally decide which one to use.
    """

    # separate_some_feature = df_fuel[['ENGINESIZE' , 'CYLINDERS' , 'FUELCONSUMPTION_COMB' , 'CO2EMISSIONS']]
    separate_some_feature = df_fuel[['ENGINESIZE' , 'CO2EMISSIONS']]

    disply(f"\nShow the important future:\n\n{separate_some_feature.head(10)}")

#---------------------------------------
    
    # By calling this function, we can graph the count of the special feature

    show_histogram(separate_some_feature)
    
#---------------------------------------

    # By calling this function, we create a graph of the relationship between the ENGINESIZE 
    # and the amount of CO2EMISSIONS
    
    show_scatter_engine_emissions(separate_some_feature)

#---------------------------------------

    # Now we need to split the ENGINESIZE to continue our modeling. It is better to use 80% of the data 
    # for model training and 20% for testing the model.

    train_X , test_X , train_y , test_y = split_train_test(separate_some_feature)


#---------------------------------------

    # After separating the data, we can build a linear regression model by calling the train_linear_model 
    # function and send the training data.

    model = train_linear_model(train_X , train_y)

#---------------------------------------

    # After building a simple linear regression model, we can now show the train and test data 
    # along with the line regression created by the model in a graph.

    show_train_test_plot(train_X , train_y , test_X , test_y , model)

#---------------------------------------
    
    # And finally, we have to calculate the cost function for this model, for this we use r2_score 
    # and also calculate the mean error of the model.

    predicted_y = model.predict(test_X)
    mean_abs_error , r2_score = calculate_metrics(test_y , predicted_y)

    disply(f"Mean Absolute Error: {mean_abs_error}")
    disply(f"R^2 Score: {r2_score}")


#---------------------------------------
if __name__ == "__main__":
    main()