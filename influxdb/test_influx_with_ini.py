from influxdb_client import InfluxDBClient

with InfluxDBClient.from_config_file("config.ini") as client:
    # use the client to access the necessary APIs
    # for example, write data using the write_api
    with client.write_api() as writer:
        writer.write(bucket="testing", record="sensor temp=23.3")