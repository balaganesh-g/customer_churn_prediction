FROM continuumio/miniconda3:4.7.12

RUN mkdir /opt/customer_churn/
ADD pyspark_project /opt/customer_churn/pyspark_project/

WORKDIR /opt/customer_churn/
ENV PYTHONPATH /opt/customer_churn
RUN python setup.py install