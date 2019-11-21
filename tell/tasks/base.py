import multiprocessing
from multiprocessing import Process

import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi

from tell.server.utils import ServerCmd, set_logger
from tell.server.zmq_decor import multi_socket


class Worker(Process):
    def __init__(self, worker_id, worker_address_list, sink_address,
                 verbose=False, **kwargs):
        super().__init__()
        self.worker_id = worker_id
        self.logger = set_logger(colored(f'WORKER-{self.worker_id}', 'yellow'),
                                 verbose)
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.n_concurrent_sockets = len(self.worker_address)
        self.sink_address = sink_address
        self.verbose = verbose
        self.is_ready = multiprocessing.Event()

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.is_ready.clear()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='n_concurrent_sockets')
    def _run(self, sink_embed, sink_token, *receivers):
        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)

        sink_embed.connect(self.sink_address)
        sink_token.connect(self.sink_address)

        self.initialize()

        for job in self.job_buffer(receivers, sink_token):
            result = self._process(job)
            message = [result['client_id'],
                       jsonapi.dumps(result['output']), ServerCmd.data_embed]

            sink_embed.send_multipart(message)
            self.logger.info(f"job done\tclient: {result['client_id']}")

    def initialize(self):
        pass

    def _process(self, msg):
        raise NotImplementedError

    def job_buffer(self, socks, sink):
        poller = zmq.Poller()
        for sock in socks:
            poller.register(sock, zmq.POLLIN)

        self.is_ready.set()
        while not self.exit_flag.is_set():
            events = dict(poller.poll())
            for sock_idx, sock in enumerate(socks):
                if sock in events:
                    client_id, raw_msg = sock.recv_multipart()
                    msg = jsonapi.loads(raw_msg)  # probably a list
                    self.logger.info(f'new job\t'
                                     f'socket: {sock_idx}\t'
                                     f'size: {len(msg)}\t'
                                     f'client: {client_id}')

                    yield {
                        'client_id': client_id,
                        'message': msg,
                    }
