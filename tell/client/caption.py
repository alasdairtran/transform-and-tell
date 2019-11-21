from overrides import overrides
from zmq.utils import jsonapi

from .base import TellClient


class CaptioningClient(TellClient):
    @overrides
    @TellClient._timeout
    def parse(self, inputs, **kwargs):
        """Parse a text.

        Overwrite this method in subclasses for different NLP tasks.
        """
        request_id = self._send(jsonapi.dumps(inputs), len(inputs))
        request_id, response = self._recv(request_id)
        client_id, output, request_id = response
        return jsonapi.loads(output)
