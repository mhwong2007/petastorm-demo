from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, ArrayType, LongType


@udf(ArrayType(FloatType()))
def assemble_features(*cols):
    return [x for x in cols]


@udf(LongType())
def transform_label(cl):
    class_to_label = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2,
    }

    return class_to_label[cl]


def transform_iris_data(data: DataFrame):
    data = (
        data
        .withColumn('features',
                    assemble_features('sepal-length', 'sepal-width', 'petal-length', 'petal-width')
                    )
        .withColumn('label', transform_label('class'))
        .select('features', 'label')
    )

    return data


def split_train_test(data: DataFrame, train_ratio: float = 0.7, test_ratio: float = 0.3):
    return data.randomSplit([train_ratio, test_ratio], seed=42)
