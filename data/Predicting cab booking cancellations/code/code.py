# encoding:utf-8
import pandas as pd

features = ["vehicle_model_id ", "package_id ", "travel_type_id "]
noneed = ["id", "user_id", "package_id", "from_city_id", "to_city_id", "to_date", "Cost_of_error"]
label_name = "Car_Cancellation"

def clearFeature(data):
    for feature in noneed:
        del data[feature]

if __name__ == '__main__':
    train_file = u"H:\Kaggle\data\Predicting cab booking cancellations\Kaggle_YourCabs_training.csv"
    data = pd.read_csv(train_file)
    clearFeature(data)
    dis = data.count()
    print "total：",dis
    print "pre:",dis/dis.max()
    data.set_index("Car_Cancellation")
    data["travel_type_id"].plot()


