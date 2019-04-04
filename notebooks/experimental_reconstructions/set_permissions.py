#!/usr/bin/env python
# usage: python set_permissions.py reconstructions
import boto3
import sys

client = boto3.client('s3')
BUCKET='motiondeblur'


sarah_id = '634b7a0686be3590c1808efc465ea9db660233386f1ad0bbbe3cabab19ae2564'
zack_id  = 'f3e01911234678438f58d65607e76c55df61e5a3db5316a30394a24767e2ceb6'


def process_s3_objects(prefix):
    """Get a list of all keys in an S3 bucket."""
    kwargs = {'Bucket': BUCKET, 'Prefix': prefix}
    failures = []
    while_true = True
    while while_true:
      resp = client.list_objects_v2(**kwargs)
      for obj in resp['Contents']:
        try:
            print(obj['Key'])
            set_acl(obj['Key'])
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            while_true = False
        except Exception:
            failures.append(obj["Key"])
            continue

    print("failures :", failures)

def set_acl(key):
  client.put_object_acl(     
    GrantFullControl="id="+sarah_id+",id="+zack_id,
    Bucket=BUCKET,
    Key=key
)

process_s3_objects(sys.argv[1])