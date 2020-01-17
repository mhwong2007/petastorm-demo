from pyspark.sql import SparkSession

from utils.reader.spark_reader import read_iris_data
from utils.transformer.spark_transformer import transform_iris_data, split_train_test


def data_transformation(
        spark: SparkSession = None,
        data_path: str = 'data/raw_data/iris.data',
        output_path: str = 'data/spark_processed_data'
):
    # initiate spark session
    if not spark:
        spark = (
            SparkSession
            .builder
            .appName('spark-data-transformation')
            .getOrCreate()
        )

    # read data
    data = read_iris_data(spark, data_path)

    # get features and label
    data = transform_iris_data(data)

    # split train test
    train, test = split_train_test(data)

    # save data
    train_path = f'{output_path}/train_iris.parquet'
    test_path = f'{output_path}/test_iris.parquet'
    (
        train
        .write
        .mode('overwrite')
        .parquet(train_path)
    )
    (
        test
        .write
        .mode('overwrite')
        .parquet(test_path)
    )
