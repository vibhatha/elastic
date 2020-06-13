import time
import redis
import unittest
from torchelastic.rendezvous.redis_server import RedisServer

class LaunchTest(unittest.TestCase):    

    @classmethod
    def setUpClass(cls):
        # start a standalone, single process etcd server to use for all tests        
       cls.redis_instance: redis.Redis = redis.Redis(host='localhost', port=6379, db=0)
       cls.redis_server : RedisServer = RedisServer() 
       print(cls.redis_server)      
       print("Setup Class")
       

    @classmethod
    def tearDownClass(cls):
        # stop the standalone etcd server
        print(cls.redis_server)
        cli: redis.Redis = cls.redis_server.get_client()
        print("Retrieving the client {}".format(cli))
        #cli.set('foo','bar')
        time.sleep(5)
        #cls.redis_server.stop()
        print("Tear Down class {}".format(cls.redis_instance))

    
    def p1(self):
        print("Test Redis")
        cli: redis.Redis = self.redis_server.get_client()
        res = cli.set('key','value')
        print("Adding Key {}".format(res))
        res = cli.get('key')
        print("Retrieving Key {}".format(res))

    def p3(self):
        print("P3")    
    
    def p2(self):
        print("P2")    

    def setUp(self):
        print("Setup Redis Server")
        self.redis_server.start()        
        

    def tearDown(self):
        print("Tear Down Redis Server")




