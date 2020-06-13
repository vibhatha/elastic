import redis
r = redis.Redis(host='localhost', port=65321, db=0)

res = r.set('foo','bar')

print(res)

res = r.get('foo')

print(res)