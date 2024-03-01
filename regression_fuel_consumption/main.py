
import numpy as np

from src.display import disply
from src.read_file import read_file
from src.description import description
from src.graphs_for_enginesize import show_histogram
from src.scatter import show_scatter_for_emissions
from src.scatter_train_test import scatter_train_test



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
    # ما اینجا تصمیم گرفتیم که با سایز موتور انجام بدیم

    # separate_some_feature = df_fuel[['ENGINESIZE' , 'CYLINDERS' , 'FUELCONSUMPTION_COMB' , 'CO2EMISSIONS']]
    separate_some_feature = df_fuel[['ENGINESIZE' , 'CO2EMISSIONS']]

    disply(f"\nShow the important future:\n\n{separate_some_feature.head(10)}")

#---------------------------------------
    
    # با این تابع فقط تعداد سایز موتور های متفاوت رو نشون میدیم
    # و بعد کامنتش میکنیم چون دیگه نمودارش رو سیو داریم

    # show_histogram(separate_some_feature['ENGINESIZE'])
    
#---------------------------------------

    # اینجا میخوایم که رابطه نموداری سایز موتور رو در مقابل تولید گاز نمایش بدیم
    
    # show_scatter_for_emissions(separate_some_feature['ENGINESIZE'] , separate_some_feature['CO2EMISSIONS'])

#---------------------------------------
    # الان تعیین کردیم که با کدوم دیتا میخوایم بریم جلو و اونم سایز موتور است  وسپس باید دیتایی که 
    # مال موتوره رو میتونیم به دو قسمت تبدیل کنیم که یعنی ۸۰ درصد رو برای اموزش دادن یا ترین جدا کنیم
    # و ۲۰ درصد رو برای تست کردن مدل و برای رندوم جدا کردن دیتاها باید از رندوم نامپای به اندازه 
    # تعداد داده هایی که دیتافریممون داره عدد رندوم بین ۰ تا ۱ تولید کنیم وسپس اعداد موچکتر از 0.8 رو
    # جدا کنیم که میشه داده های ترین ما و بقیه هم میشن داده های تست ما

    msk = np.random.rand(len(df_fuel)) < 0.8

    train = separate_some_feature[msk]  # حدود ۸۰ درصد رو به عنوان ترین جدا میکنه برامون
    test = separate_some_feature[~msk]  # و با علامت تیلدا بقیه رو به عنوان تست جدا میکنه

    disply(f"\nshow the train data:\n\n{train}")
    disply(f"\nshow the test data:\n\n{test}")

#---------------------------------------
    
    # دوباره نمودار رابطه رو رسم میکنیم اما اینبار به جای کل دیتا دیتای ترین رو با رنگ جدا و تست رو با رنگ جدا
    # حالا ما سعی میکنیم که از داده ترین بهترین خطی که ممکنه رو برای پیش بینی پیدا کنیم و با داده تست
    # دقتش رو بررسی کنیم و ببینیم که چقد درست پیش بینی میکنه

    # scatter_train_test(train , test)

#---------------------------------------

    


#---------------------------------------
if __name__ == "__main__":
    main()