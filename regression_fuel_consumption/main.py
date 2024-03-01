

from src.display import disply
from src.read_file import read_file
from src.description import description , show_info_data
from src.count_engine_size import show_histogram
from src.scatter_engine_emission import show_scatter_engine_emissions
from src.split_train_test import split_train_test
from src.train_linear_model import train_linear_model
from src.show_train_test_plot import show_train_test_plot


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

    # اینجا متوجه میشیم که هیچ نالی نداره
    show_info_data(df_fuel)
    print(100*"*")

#---------------------------------------


    # جدا کردن بعضی ویژگی های مهم که میخوایم روشون کار کنیم و چون رگرشن ساده داریم با یکی از این
    # فیچرها کار میکنیم که باید بررسی کنیم ببینیم کدوم بهتر برای مدلمون کار میکنه
    # ما اینجا تصمیم گرفتیم که با سایز موتور انجام بدیم

    # separate_some_feature = df_fuel[['ENGINESIZE' , 'CYLINDERS' , 'FUELCONSUMPTION_COMB' , 'CO2EMISSIONS']]
    separate_some_feature = df_fuel[['ENGINESIZE' , 'CO2EMISSIONS']]

    disply(f"\nShow the important future:\n\n{separate_some_feature.head(10)}")

#---------------------------------------
    
    # با این تابع فقط تعداد سایز موتور های متفاوت رو نشون میدیم
    # و بعد کامنتش میکنیم چون دیگه نمودارش رو سیو داریم

    # show_histogram(separate_some_feature)
    
#---------------------------------------

    # اینجا میخوایم که رابطه نموداری سایز موتور رو در مقابل تولید گاز نمایش بدیم
    
    # show_scatter_engine_emissions(separate_some_feature)

#---------------------------------------
    # الان تعیین کردیم که با کدوم دیتا میخوایم بریم جلو و اونم سایز موتور است  وسپس باید دیتایی که 
    # مال موتوره رو میتونیم به دو قسمت تبدیل کنیم که یعنی ۸۰ درصد رو برای اموزش دادن یا ترین جدا کنیم
    # ۲۰ درصد رو هم برای تست

    train_X , test_X , train_y , test_y = split_train_test(separate_some_feature)


#---------------------------------------
    # حالا که داده های ترین و تست رو داریم میتونیم بریم و از سایکیلرن مدل خطیمون رو بسازیم
    # ما ترین ایکس رو اینجین سایز قرار میدیم و ترین وای رو تولید گاز و توی مدلمون باید فیت بشه
    # و توی این تابعی که صدا میزنیم کوفیشنت و اینترسفت رو حساب میکنیم که میشه نقطه شروع نمودار و ضریب ایکس
    # که اینجا ضریب ایکس میشه ضریب سایز موتور


    model = train_linear_model(train_X , train_y)

#---------------------------------------


    show_train_test_plot(train_X , train_y , test_X , test_y , model)



#---------------------------------------
if __name__ == "__main__":
    main()