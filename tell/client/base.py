import uuid
from collections import namedtuple
from functools import wraps

import zmq
from zmq.utils import jsonapi

# Client version must match server version
__version__ = '0.0.1'

_Response = namedtuple('_Response', ['id', 'content'])


class TellClient:
    def __init__(self, ip='localhost', port=5555, port_out=5556, identity=None,
                 ignore_checks=False, timeout=-1, verbose=False):
        """Create a client for the Tell Server.

        This creates a client that connects to a Tell Server. The server must
        be ready at the moment we call initialize this object. If we're not
        sure if the server is ready or not, set ignore_checks to True.

        Parameters
        ----------
        ip : str
            The IP address of the server.
        port: int
            The port to push data from client to server.
        port_out: int
            The port for receiving results from server.
        identity: str
            The UUID of this client.
        ignore_checks : bool
            Set this to True if we're unsure if the server is ready now.
        timeout : int
            Set the timeout (in milliseconds) for receiving operation on the
            client. -1 means no timeout and wait until result returns.

        Examples
        --------
        We can use this as a context manager, which automatically closes the
        client once we're done parsing
            with TellClient() as tc:
                tc.parse(text)
        """
        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PUSH)
        # Don't linger around when the socket has been closed
        self.sender.setsockopt(zmq.LINGER, 0)
        self.identity = identity or str(uuid.uuid4()).encode('ascii')
        self.sender.connect(f'tcp://{ip}:{port}')

        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.LINGER, 0)
        # Subscribe to all messages beginning with the client UUID. When we
        # receive a message from the publisher, we'll check that the first
        # element of the message matches our UUID.
        self.receiver.setsockopt(zmq.SUBSCRIBE, self.identity)
        self.receiver.connect(f'tcp://{ip}:{port_out}')

        # Increment this for every new request
        self.request_id = 0
        self.timeout = timeout
        self.pending_request = set()
        # When we receive responses out-of-order, store them in a buffer
        self.pending_response = {}
        self.port = port
        self.port_out = port_out
        self.ip = ip

        if not ignore_checks:
            sever_status = self.server_status
            server_version = sever_status['server_version']
            client_version = self.status['client_version']
            if server_version != client_version:
                raise AttributeError(f'Version mismatch! server version is '
                                     f'{server_version} but client version is '
                                     f'{client_version}.')
            if verbose:
                self._print_dict(sever_status, 'Server config:')

    def close(self):
        """Close all connections of the client gracefully.

        This is automatically called if we use the context manager syntax.
        """
        self.sender.close()
        self.receiver.close()
        self.context.term()

    def _send(self, msg, msg_len=0):
        self.request_id += 1
        self.sender.send_multipart(
            [self.identity, msg, b'%d' % self.request_id, b'%d' % msg_len])
        self.pending_request.add(self.request_id)
        return self.request_id

    def _recv(self, wait_for_req_id=None):
        try:
            while True:
                # a request has been returned and found in pending_response
                if wait_for_req_id in self.pending_response:
                    response = self.pending_response.pop(wait_for_req_id)
                    return _Response(wait_for_req_id, response)

                # receive a response
                response = self.receiver.recv_multipart()
                request_id = int(response[-1])

                # if not wait for particular response then simply return
                if not wait_for_req_id or (wait_for_req_id == request_id):
                    self.pending_request.remove(request_id)
                    return _Response(request_id, response)
                elif wait_for_req_id != request_id:
                    self.pending_response[request_id] = response
                    # wait for the next response
        except Exception as e:
            raise e
        finally:
            if wait_for_req_id in self.pending_request:
                self.pending_request.remove(wait_for_req_id)

    def _timeout(func):  # pylint: disable=no-self-argument
        @wraps(func)
        def arg_wrapper(self, *args, **kwargs):
            if 'blocking' in kwargs and not kwargs['blocking']:
                # override client timeout setting if `func` is called in non-blocking way
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)
            else:
                self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
            try:
                return func(self, *args, **kwargs)
            except zmq.error.Again as _e:
                t_e = TimeoutError(
                    f'No response from the server (with "timeout"='
                    f'{self.timeout} ms), please check the following: Is the '
                    f'server still online? Is the network broken? Are "port" '
                    f'and "port_out" correct? Are you encoding a huge amount '
                    f'of data whereas the timeout is too small for that?')
                raise t_e from _e
            finally:
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)

        return arg_wrapper

    @property
    def status(self):
        """Get the status of this Tell Client instance.

        Returns
        -------
        status : Dict[str, str]
            A dictionary containing the current status of this Tell Client
            instance.
        """
        return {
            'identity': self.identity,
            'num_request': self.request_id,
            'num_pending_request': len(self.pending_request),
            'pending_request': self.pending_request,
            'port': self.port,
            'port_out': self.port_out,
            'server_ip': self.ip,
            'client_version': __version__,
            'timeout': self.timeout
        }

    @property  # type: ignore
    @_timeout
    def server_status(self):
        """Get the current status of the server connected to this client.

        Returns
        -------
        status : Dict[str, str]
            A dictionary containing the current status of the server connected
            to this client.
        """
        req_id = self._send(b'SHOW_CONFIG')
        return jsonapi.loads(self._recv(req_id).content[1])

    @_timeout
    def parse(self, texts, blocking=True, **kwargs):
        """Parse a text.

        Overwrite this method in subclasses for different NLP tasks.
        """
        request_id = self._send(jsonapi.dumps(texts), len(texts))
        if not blocking:
            return request_id
        request_id, response = self._recv(request_id)
        client_id, output, request_id = response
        return jsonapi.loads(output)

    @_timeout
    def fetch(self, request_id):
        request_id, response = self._recv(request_id)
        client_id, output, request_id = response
        return jsonapi.loads(output)

    @staticmethod
    def _print_dict(x, title=None):
        if title:
            print(title)
        for k, v in x.items():
            print('%30s\t=\t%-30s' % (k, v))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
