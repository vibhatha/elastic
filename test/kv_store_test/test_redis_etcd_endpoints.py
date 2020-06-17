import redis
import etcd


etcd_endpoints = (("localhost", 2379),)
etcd_client: etcd.Client = etcd.Client(
    host=etcd_endpoints, allow_reconnect=True)

redis_client: redis.Redis = redis.Redis(host='localhost', port=49770, db=0)

print(etcd_client, redis_client)

full_path = "test/"
ttl = None

def test_create_key_type1(run=False):
    if run:
        etcd_client.write(
                    key=full_path, value=None, dir=True, prevExist=False, ttl=ttl
        )
        redis_client.set(full_path, "", nx=True, ex=ttl, keepttl=False)

def test_get_key_type1(run=False):
    redis_val = None
    etcd_val = None
    if run:
        redis_val = redis_client.get(full_path)
        try:
            etcd_val = etcd_client.get(full_path)
        except etcd.EtcdKeyNotFound:
            print("Etcd Key {} not found".format(full_path)) 
        finally:
            print("ETCD")
            print(etcd_val)
            print("REDIS")
            print(redis_val)

def test_remove_keys(key, run=False, is_dir=False):
    if run:
        etcd_client.delete(key, dir=is_dir)
        redis_client.delete(key)

def etcd_clients():
    return etcd_client.machines


def remove_key(key=None):
    if key is None:
        raise ValueError("Key cannot be none")
    redis_client.delete(key)
    try:
        etcd_client.delete(key)
    except etcd.EtcdNotFile:
        etcd_client.delete(key,dir=True)    

def test_create_key_type2(run=False, value=None, prev_exists=False, dir=False):
    # Previoius Existence Check
    if run:
        try:
            etcd_client.write(
                        key=full_path, value=value, dir=dir, prevExist=prev_exists, ttl=ttl
            )
        except etcd.EtcdAlreadyExist:
            print("ETCD Key {} already exists".format(full_path))    
        if value is None:
            value = ""
        redis_client.set(full_path, value, nx=not prev_exists, ex=ttl, keepttl=False)

        etcd_res = etcd_client.get(full_path)
        redis_res = redis_client.get(full_path)
        print("ETCD : {}".format(etcd_res))
        print("REDIS : {}".format(redis_res))

def create_key_value(key=None, value=None,):
    if key is None or value is None:
        raise ValueError("Key or value is None")
    etcd_client.write(
                key=key, 
                value=value)

    redis_client.set(key, 
                    value)

def get_key_value(key=None):
    if key is None:
        raise ValueError("Key cannot be None")

    return (etcd_client.get(key), redis_client.get(key))

def redis_test_and_set(redis_client: redis.Redis, key=None, new_value=None, prev_value=None):            
    response = None
    if key is None:
        raise ValueError("Key cannot be None")
    if prev_value == new_value:        
        response = redis_client.set(key, new_value)
    else:
        # TODO: Check if the else condition has to handled or just avoided     
        pass
    return response

def test_and_create_compare(key=None, value=None, prev_value=None):
    if key is None or value is None:
        raise ValueError("Key or Value is None")
    
    etcd_response = etcd_client.test_and_set(
                    key=key,
                    value=value,
                    prev_value=prev_value,
                    ttl=None,
                )
    redis_response = redis_test_and_set(redis_client=redis_client,
                     key=key,
                     value=value,
                     prev_value=prev_value)      

    etcd_result = etcd_client.get(key)
    redis_result = redis_client.get(key) 

    return etcd_response, etcd_result, redis_response, redis_result                       
    

def test_test_and_create_compare(run=False):
    if run:
        new_key="key1"
        new_value="value1"
        prev_value="value0"

        create_key_value(key=new_key, value=prev_value)
        (etcd_result, redis_result) = get_key_value(key=new_key)

        print("ETCD")
        print(etcd_result)
        print("REDIS")
        print(redis_result)

        if etcd_result.value == prev_value and redis_result == prev_value:
            etcd_test_and_set_result = etcd_client.test_and_set(
                                        key=new_key,
                                        value=new_value,
                                        prev_value=prev_value
                                        )


            redis_test_and_set_result = redis_test_and_set(
                                        redis_client, 
                                        key=new_key, 
                                        new_value=new_value, 
                                        prev_value=prev_value
            )
            print("ETCD")
            print(etcd_test_and_set_result)
            print("REDIS")
            print(redis_test_and_set_result)

#test_create_key_type1(run=False)
#test_get_key_type1(run=False)
#test_remove_keys(key=full_path, run=False, is_dir=False)
#remove_key(key=full_path)
#test_create_key_type2(run=True, value="14", prev_exists=False)


test_test_and_create_compare(run=True)



