import random
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict

import numpy as np
import zmq
import zmq.decorators as zmqd
from termcolor import colored
from torch.multiprocessing import Event, Process, set_start_method
from zmq.utils import jsonapi

from tell.tasks import WorkerRegistry

from .utils import ServerCmd, auto_bind, set_logger
from .zmq_decor import multi_socket

__version__ = '0.0.1'

# See https://stackoverflow.com/a/48938860
try:
    set_start_method('spawn')
except RuntimeError:
    pass


class NLPServer(threading.Thread):
    """For connecting two processes in the same server it is considered that IPC is the fastest option"""

    def __init__(self, port=5558, port_out=5559, n_workers=1, verbose=False,
                 max_batch_size=32, task='coref'):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), verbose)
        self.port = port
        self.port_out = port_out
        self.processes = []
        self.is_ready = threading.Event()
        self.n_workers = n_workers
        self.n_concurrent_sockets = max(8, n_workers * 2)
        self.max_batch_size = max_batch_size
        self.status_static = {
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }
        self.Worker = WorkerRegistry[task]

    def __enter__(self):
        self.start()
        self.is_ready.wait()
        return self

    def __exit__(self,  exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        self.is_ready.clear()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, frontend):
        frontend.connect('tcp://localhost:%d' % self.port)
        frontend.send_multipart([b'', ServerCmd.terminate, b'', b''])

    @staticmethod
    def shutdown(args):
        with zmq.Context() as ctx:
            ctx.setsockopt(zmq.LINGER, args.timeout)
            with ctx.socket(zmq.PUSH) as frontend:
                try:
                    frontend.connect('tcp://%s:%d' % (args.ip, args.port))
                    frontend.send_multipart(
                        [b'', ServerCmd.terminate, b'', b''])
                    print('shutdown signal sent to %d' % args.port)
                except zmq.error.Again:
                    raise TimeoutError(
                        'no response from the server (with "timeout"=%d ms), please check the following:'
                        'is the server still online? is the network broken? are "port" correct? ' % args.timeout)

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_socket(zmq.PUSH, num_socket='n_concurrent_sockets')
    def _run(self, _, frontend, sink, *backend_socks):

        def push_new_job(_job_id, _json_msg, _msg_len):
            _sock = rand_backend_socket
            _sock.send_multipart([_job_id, _json_msg])

        self.logger.info(f'Bind all sockets. Use ports '
                         f'{self.port}/{self.port_out}')
        frontend.bind(f'tcp://*:{self.port}')
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info(f'open {len(addr_backend_list)} ventilator-worker '
                         'sockets')

        self.logger.info('Start the sink')
        proc_sink = Sink(self.port_out, addr_front2sink)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start the backend processes
        device_map = [-1] * self.n_workers
        for idx, device_id in enumerate(device_map):
            process = self.Worker(idx, addr_backend_list, addr_sink)
            self.processes.append(process)
            process.start()

        rand_backend_socket = None
        server_status = ServerStatistic()

        for p in self.processes:
            p.is_ready.wait()

        self.is_ready.set()
        self.logger.info('all set, ready to serve request!')

        while True:
            try:
                request = frontend.recv_multipart()
                client, msg, req_id, msg_len = request
            except ValueError:
                self.logger.error(
                    'received a wrongly-formatted request (expected 4 frames, got %d)' % len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k)
                                            for idx, k in enumerate(request)), exc_info=True)
            else:
                server_status.update(request)
                if msg == ServerCmd.terminate:
                    break
                elif msg == ServerCmd.show_config:
                    self.logger.info(
                        'new config request\treq id: %d\tclient: %s' % (int(req_id), client))
                    status_runtime = {'client': client.decode('ascii'),
                                      'num_process': len(self.processes),
                                      'ventilator -> worker': addr_backend_list,
                                      'worker -> sink': addr_sink,
                                      'ventilator <-> sink': addr_front2sink,
                                      'server_current_time': str(datetime.now()),
                                      'statistic': server_status.value,
                                      'device_map': device_map,
                                      'n_concurrent_sockets': self.n_concurrent_sockets}

                    sink.send_multipart([client, msg, jsonapi.dumps({**status_runtime,
                                                                     **self.status_static}), req_id])
                else:
                    self.logger.info('new encode request\treq id: %d\tsize: %d\tclient: %s' %
                                     (int(req_id), int(msg_len), client))
                    # register a new job at sink
                    sink.send_multipart(
                        [client, ServerCmd.new_job, msg_len, req_id])

                    # renew the backend socket to prevent large job queueing up
                    # [0] is reserved for high priority job
                    # last used backennd shouldn't be selected either as it may be queued up already
                    rand_backend_socket = random.choice(
                        [b for b in backend_socks[1:] if b != rand_backend_socket])

                    # push a new job, note super large job will be pushed to one socket only,
                    # leaving other sockets free
                    job_id = client + b'#' + req_id
                    if int(msg_len) > self.max_batch_size:
                        seqs = jsonapi.loads(msg)
                        job_gen = []
                        for i in range(0, int(msg_len), self.max_batch_size):
                            pid = job_id + b'@%d' % i
                            pjob = seqs[i:(i + self.max_batch_size)]
                            job_gen.append((pid, pjob))

                        for partial_job_id, job in job_gen:
                            push_new_job(partial_job_id,
                                         jsonapi.dumps(job), len(job))
                    else:
                        push_new_job(job_id, msg, int(msg_len))

        for p in self.processes:
            p.close()
        self.logger.info('terminated!')


class Sink(Process):
    def __init__(self, port_out, front_sink_addr, verbose=False):
        super().__init__()
        self.port = port_out
        self.exit_flag = Event()
        self.logger = set_logger(colored('SINK', 'green'), verbose)
        self.front_sink_addr = front_sink_addr
        self.is_ready = Event()
        self.verbose = verbose

    def close(self):
        self.logger.info('shutting down...')
        self.is_ready.clear()
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        pending_jobs: Dict[str, SinkJob] = defaultdict(lambda: SinkJob())

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver_addr.encode('ascii'))

        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compability
        logger = set_logger(colored('SINK', 'green'), self.verbose)
        logger.info('ready')
        self.is_ready.set()

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            if socks.get(receiver) == zmq.POLLIN:
                msg = receiver.recv_multipart()
                job_id = msg[0]
                # parsing job_id and partial_id
                job_info = job_id.split(b'@')
                job_id = job_info[0]
                partial_id = int(job_info[1]) if len(job_info) == 2 else 0

                if msg[2] == ServerCmd.data_embed:
                    x = jsonapi.loads(msg[1])
                    pending_jobs[job_id].add_output(x, partial_id)
                else:
                    logger.error(
                        'received a wrongly-formatted request (expected 4 frames, got %d)' % len(msg))
                    logger.error('\n'.join('field %d: %s' % (idx, k)
                                           for idx, k in enumerate(msg)), exc_info=True)

                logger.info('collect %s %s (E:%d/A:%d)' % (msg[2], job_id,
                                                           pending_jobs[job_id].progress_outputs,
                                                           pending_jobs[job_id].checksum))

                # check if there are finished jobs, then send it back to workers

                finished = [(k, v)
                            for k, v in pending_jobs.items() if v.is_done]
                for job_info, tmp in finished:
                    client_addr, req_id = job_info.split(b'#')
                    x = tmp.result
                    sender.send_multipart([client_addr, x, req_id])
                    logger.info('send back\tsize: %d\tjob id: %s' %
                                (tmp.checksum, job_info))
                    # release the job
                    tmp.clear()
                    pending_jobs.pop(job_info)

            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                if msg_type == ServerCmd.new_job:
                    job_info = client_addr + b'#' + req_id
                    # register a new job
                    pending_jobs[job_info].checksum = int(msg_info)
                    logger.info('job register\tsize: %d\tjob id: %s' %
                                (int(msg_info), job_info))
                elif msg_type == ServerCmd.show_config:
                    # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    time.sleep(0.1)
                    logger.info('send config\tclient %s' % client_addr)
                    sender.send_multipart([client_addr, msg_info, req_id])


class SinkJob:
    def __init__(self):
        self.outputs = []
        self.output_ids = []
        self.checksum = 0  # message length
        self.progress_outputs = 0

    def clear(self):
        self.outputs.clear()

    def add_output(self, data, pid):
        progress = len(data)
        self.outputs.append(data)
        self.output_ids.append(pid)
        self.progress_outputs += progress

    @property
    def is_done(self):
        return self.checksum > 0 and self.checksum == self.progress_outputs

    @property
    def result(self):
        # Sort the results
        sort_idx = np.argsort(self.output_ids)
        outputs = np.array(self.outputs)[sort_idx].tolist()
        outputs = [elem for output in outputs for elem in output]
        return jsonapi.dumps(outputs)


class ServerStatistic:
    def __init__(self):
        self._hist_client = defaultdict(int)
        self._hist_msg_len = defaultdict(int)
        self._client_last_active_time = defaultdict(float)
        self._num_data_req = 0
        self._num_sys_req = 0
        self._num_total_seq = 0
        self._last_req_time = time.perf_counter()
        self._last_two_req_interval = []
        self._num_last_two_req = 200

    def update(self, request):
        client, msg, req_id, msg_len = request
        self._hist_client[client] += 1
        if ServerCmd.is_valid(msg):
            self._num_sys_req += 1
            # do not count for system request, as they are mainly for heartbeats
        else:
            self._hist_msg_len[int(msg_len)] += 1
            self._num_total_seq += int(msg_len)
            self._num_data_req += 1
            tmp = time.perf_counter()
            self._client_last_active_time[client] = tmp
            if len(self._last_two_req_interval) < self._num_last_two_req:
                self._last_two_req_interval.append(tmp - self._last_req_time)
            else:
                self._last_two_req_interval.pop(0)
            self._last_req_time = tmp

    @property
    def value(self):
        def get_min_max_avg(name, stat):
            if len(stat) > 0:
                return {
                    'avg_%s' % name: sum(stat) / len(stat),
                    'min_%s' % name: min(stat),
                    'max_%s' % name: max(stat),
                    'num_min_%s' % name: sum(v == min(stat) for v in stat),
                    'num_max_%s' % name: sum(v == max(stat) for v in stat),
                }
            else:
                return {}

        def get_num_active_client(interval=180):
            # we count a client active when its last request is within 3 min.
            now = time.perf_counter()
            return sum(1 for v in self._client_last_active_time.values() if (now - v) < interval)

        parts = [{
            'num_data_request': self._num_data_req,
            'num_total_seq': self._num_total_seq,
            'num_sys_request': self._num_sys_req,
            'num_total_request': self._num_data_req + self._num_sys_req,
            'num_total_client': len(self._hist_client),
            'num_active_client': get_num_active_client()},
            get_min_max_avg('request_per_client', self._hist_client.values()),
            get_min_max_avg('size_per_request', self._hist_msg_len.keys()),
            get_min_max_avg('last_two_interval', self._last_two_req_interval),
            get_min_max_avg('request_per_second', [
                            1. / v for v in self._last_two_req_interval]),
        ]

        return {k: v for d in parts for k, v in d.items()}
