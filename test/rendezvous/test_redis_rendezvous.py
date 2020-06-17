import re
import redis
from urllib.parse import urlparse
url = "etcd://localhost:6379/123?min_workers=4&max_workers=8&timeout=300"



url = urlparse(url)
assert url.scheme == "etcd"

print(url.scheme)
print(url.netloc)

endpoint = url.netloc

match = re.match(r"(.+):(\d+)$", endpoint)  # check if port was provided
if match:
    etcd_endpoints = ((match.group(1), int(match.group(2))),)
else:
    # Use default etcd port
    etcd_endpoints = ((endpoint, 2379),)

print(etcd_endpoints[0][0], etcd_endpoints[0][1])
host = etcd_endpoints[0][0]
port = etcd_endpoints[0][1]

r = redis.Redis(host=host, port=port)

r.set("/nodes/n1","1")
r.set("/nodes/n2","2")
r.setnx("/nodes/n3/sub","")
r.set("/nodes/n3/sub","5")

#print(r.get("/nodes/n3/sub"))


print(r.delete("/nodes/n4/sub"))
import datetime
t1 = datetime.timedelta(seconds=3)
print(t1, type(t1), t1.total_seconds())
#r.set("/nodes/n4/sub","",px=t1,nx=True,keepttl=True)
#r.set("/nodes/n4/sub","",nx=True,ex=None,keepttl=False)
#r.set("/nodes/n4/sub","3")
#r.set("/nodes/n4/sub","",nx=True)
r.set("/nodes/n4/sub","6",nx=True,ex=None,keepttl=False)
r.set("/nodes/n4/sub","1",xx=False)
d = r.get("/nodes/n4/sub")
print("check nx: ",d, type(d))

r.lpush("/nodes/n5/sub",3)

len1 = r.llen('/nodes/n5/sub')

val = r.lindex('/nodes/n5/sub', 0)

print("Len {}".format(len1))

print("Retrived Index Val {}".format(val))

r.zadd()





