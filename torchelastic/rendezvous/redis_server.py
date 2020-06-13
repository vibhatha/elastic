import os
import time
import tempfile
import redis
import logging
import socket
import shlex
import shutil
import atexit
import subprocess


log = logging.getLogger(__name__)

def find_free_port():
    """
    Finds a free port and binds a temporary socket to it so that
    the port can be "reserved" until used.

    .. note:: the returned socket must be closed before using the port,
              otherwise a ``address already in use`` error will happen.
              The socket should be held and closed as close to the
              consumer of the port as possible since otherwise, there
              is a greater chance of race-condition where a different
              process may see the port as being free and take it.

    Returns: a socket binded to the reserved free port

    Usage::

    sock = find_free_port()
    port = sock.getsockname()[1]
    sock.close()
    use_port(port)
    """
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )

    for addr in addrs:
        family, type, proto, _, _ = addr
        try:
            s = socket.socket(family, type, proto)
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
            print("Socket creation attempt failed: " + e)
    raise RuntimeError("Failed to create a socket")


def _prepare_redis_conf(redis_conf_path: str, port: int):
    # /etc/redis/6379.conf
    # WARNING: make sure the write previliges are there for the redis_conf_path
    # port              6379
    # daemonize         yes
    # save              60 1
    # bind              127.0.0.1
    # tcp-keepalive     300
    # dbfilename        dump.rdb
    # dir               ./
    # rdbcompression    yes
    redis_conf = [
                    "port \t\t\t {}".format(port),
                    "daemonize \t\t yes",
                    "save \t\t\t 60 1",
                    "bind \t\t\t 127.0.0.1",
                    "tcp-keepalive  \t\t 300",
                    "dbfilename \t\t dump.rdb",
                    "dir \t\t\t ./",
                    "rdbcompression \t\t yes",                    
                ]
    
    
    with open(redis_conf_path, "w+") as fp:
        for line in redis_conf:
            fp.write(line + "\n")
    
    #     log.info("Redis Config Already Exists!")
    #     with open(redis_conf_path, "r") as fp:
    #         lines = fp.readlines()            
    #         print(lines)
            
    print("Redis Config {}".format(redis_conf))     


def stop_redis(subprocess_instance, data_dir, port, host):
    if subprocess_instance and subprocess_instance.poll() is None:
        log.info(f"stopping redis server")
        print("stopping redis server")
        subprocess_instance.terminate()
        subprocess_instance.wait()
    elif subprocess is not None:
         #redis-cli -h localhost -p 65321 shutdown
        redis_cmd = shlex.split(
            " ".join(
                [
                    "redis-cli",
                    "-h",
                    str(host),
                    "-p",
                    str(port),
                    "shutdown"
                ]
            )
        )         
        log.info("Redis server stop cmd : {}".format(redis_cmd))
        _redis_kill_proc = subprocess.Popen(redis_cmd, close_fds=True)         
    else:
        print("Subproces instance : {}".format(subprocess_instance.poll()))    
        print("Subprocess {}".format(subprocess_instance))

    log.info(f"deleting redis data dir: {data_dir}")
    print(f"deleting redis data dir: {data_dir}")
    shutil.rmtree(data_dir, ignore_errors=True)


class RedisServer(object):

    def __init__(self):
        self._port = -1
        self._host = "localhost"

        root = os.path.dirname(__file__)
        default_redis_bin = os.path.join(root, "bin/redis-server")
        default_redis_conf = "/etc/redis/default.conf"
        
        self._redis_conf_path = os.environ.get(
            "TORCHELASTIC_REDIS_CONF_PATH", default_redis_conf
        )
        self._redis_binary_path = os.environ.get(
            "TORCHELASTIC_REDIS_BINARY_PATH", default_redis_bin
        )

        if not os.path.isfile(self._redis_binary_path):
            self._redis_binary_path = "redis-server"

        self._data_dir = tempfile.mkdtemp(prefix="torchelastic_redis_data")
        self._redis_cmd = None
        self._redis_proc = None

    def get_port(self) -> int:
        """
        Returns:
            the port the server is running on.
        """
        return self._port

    def get_host(self) -> str:
        """
        Returns:
            the host the server is running on.
        """
        return self._host

    def get_endpoint(self) -> str:
        """
        Returns:
            the redis server endpoint (host:port)
        """
        return f"{self._host}:{self._port}"

    def start(self, timeout: int = 60) -> None:        
        log.info("Start Redis Server")
        sock = find_free_port()        
        self._port = sock.getsockname()[1]
        sock.close()

        _prepare_redis_conf(redis_conf_path=self._redis_conf_path, port=self._port)

        redis_cmd = shlex.split(
            " ".join(
                [
                    self._redis_binary_path,
                    #"--port",
                    #str(self._port),
                    self._redis_conf_path,
                ]
            )
        ) 
        print(redis_cmd)       
        log.info("Redis launcher cmd : {}".format(redis_cmd))
        self._redis_proc = subprocess.Popen(redis_cmd, close_fds=True)
        atexit.register(stop_redis, self._redis_proc, self._data_dir, self._port, self._host)
        self._wait_for_ready(timeout)
        print("Redis Server with PID {}".format(self._redis_proc))
        

    def get_client(self) -> redis.Redis:
        return redis.Redis(
            host=self._host, port=self._port, db=0
            )

    def _wait_for_ready(self, timeout: int = 60):
        client = redis.Redis(self._host, port=self._port, db=0)
        max_time = time.time() + timeout
        count = 1 
        while time.time() < max_time:
            try:
                log.info(f"Redis server ready : {client.ping()}")
                print(f"Redis server ready : {client.ping()}")
                return
            except Exception as e:
                time.sleep(1)
                print("Failure, Re-attempt : {} => error {}".format(count, e.__str__()))
            count = count + 1    
        raise TimeoutError("Timed out waiting for etcd server to be ready!")

    def stop(self) -> None:
        log.info("Stopping Redis Server : {}".format(self.__str__()))
        stop_redis(self._redis_proc, self._data_dir, self._port, self._host)


    def __str__(self):
        line: str  = "------------------------------------------------------------\n" 
        to_str: str = 'Redis Server Config\n' + line 
        to_str += "port="+ str(self._port) + "\n" \
             + "redis_exec="+self._redis_binary_path + "\n" \
             + "redis_host="+self._host + "\n" \
             + "redis_data_dir="+self._data_dir + "\n" \
             + line 
        return to_str
    
    def __repr__(self):
        return self.__str__()
