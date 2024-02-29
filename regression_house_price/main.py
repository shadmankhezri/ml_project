

from src.display import disply
from src.read_file import read_file
from src.description import description
from src.histogram_for_ssf import show_histogram
import matplotlib.pyplot as plt


#---------------------------------------

PATH = "data/FuelConsumption.csv"


#---------------------------------------

def main():
    
    df_fuel = read_file(PATH)
    disply(f"\nShow the first data frame of Fuel Consumption:\n\n{df_fuel}")

#---------------------------------------

    # اینجا متوجه میشیم که بعضی از فیچر ها رو لازم نداریم چون مثلا همه سالشون یکیه و ...ا
    df_description = description(df_fuel)
    disply(f"\nShow some information about data frame:\n\n{df_description}")

#---------------------------------------


    # جدا کردن بعضی ویژگی های مهم که میخوایم روشون کار کنیم و چون رگرشن ساده داریم با یکی از این
    # فیچرها کار میکنیم که باید بررسی کنیم ببینیم کدوم بهتر برای مدلمون کار میکنه

    separate_some_feature = df_fuel[['ENGINESIZE' , 'CYLINDERS' , 'FUELCONSUMPTION_COMB' , 'CO2EMISSIONS']]
    # separate_some_feature = df_fuel[['ENGINESIZE']]

    disply(f"\nShow the important future:\n\n{separate_some_feature.head(10)}")

#---------------------------------------
    
    for i in separate_some_feature:

        show_histogram([[i]] , 'i')
        break

#---------------------------------------

if __name__ == "__main__":
    main()