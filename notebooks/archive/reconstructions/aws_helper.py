import boto3
import datetime
import base64
import numpy as np
import botocore
import paramiko
import time


def start_instance(ami, price, keyname, instance_type, start_script=""):
    start_script = base64.b64encode(start_script.encode("ascii")).decode('ascii')
    launch_specification={
    "NetworkInterfaces": [
        {
            "AssociatePublicIpAddress": True,
            "DeleteOnTermination": True,
            "Description": "",
            "DeviceIndex": 0,
            "Groups": [
                "sg-0f5165083df8e479f"
            ],
            "Ipv6Addresses": [],
        }
    ],
    "ImageId": ami,
    "InstanceType": instance_type,
    "KeyName": keyname,
    'UserData': start_script,
    "Monitoring": {
        "Enabled": True
    },
    "Placement": {
        "AvailabilityZone": "us-west-2c",
        "GroupName": "",
        "Tenancy": "default"
    } }
    client = boto3.client('ec2', region_name='us-west-2')
    response = client.request_spot_instances(
        DryRun=False,
        SpotPrice=price,
        InstanceCount=1,
        Type='one-time',
        LaunchSpecification=launch_specification)

    request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
    print("Wait for request to be fulfilled...")
    waiter = client.get_waiter('spot_instance_request_fulfilled')
    waiter.wait(SpotInstanceRequestIds=[request_id])
    print("Wait for ec2 instance to boot up...")
    time.sleep(2)

    res2 = client.describe_spot_instance_requests()
    idx = np.argmax([res2['SpotInstanceRequests'][i]['CreateTime'] for i in range(len(res2['SpotInstanceRequests']))])
    instance_id = res2['SpotInstanceRequests'][idx]['InstanceId']

    public_dns = client.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]['PublicDnsName']

    return instance_id, public_dns

def get_all_instances():
    client = boto3.client('ec2')
    descriptions = client.describe_instances()
    len_res = len(descriptions['Reservations'])
    ret = []
    for i in range(len_res):
        len_inst = len(descriptions['Reservations'][i]['Instances'])
        ret.append([descriptions['Reservations'][i]['Instances'][j]['PublicDnsName'] for j in range(len_inst)])
    return ret


def send_command_to_instance(cmd, instance_address, keypath):
    key = paramiko.RSAKey.from_private_key_file(keypath)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(hostname=instance_address, username="ubuntu", pkey=key)

    stdin, stdout, stderr = client.exec_command(cmd)
    print(stdout.read().decode("utf-8"))
    print(stderr.read().decode("utf-8"))
    # close the client connection once the job is done
    client.close()
