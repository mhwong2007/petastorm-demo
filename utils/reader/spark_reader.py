from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, FloatType, StringType


def read_iris_data(spark: SparkSession, path: str = 'data/raw_data/iris.data'):
    schema = StructType([
        StructField('sepal-length', FloatType(), True),
        StructField('sepal-width', FloatType(), True),
        StructField('petal-length', FloatType(), True),
        StructField('petal-width', FloatType(), True),
        StructField('class', StringType(), True),
    ])

    data = (
        spark
        .read
        .schema(schema)
        .csv(path)
    )

    return data
