from job.spark_job import data_transformation
from job.train_job import train_job

if __name__ == '__main__':
    # spark data transformation
    data_transformation()

    # train model
    train_job()
