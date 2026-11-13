import os
import socket
import sys

import pytest

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    original_connect=socket.socket.connect
    def blocked(*args, **kwargs):
        # Python 3.12 on Windows builds asyncio's wake-up socketpair over
        # loopback TCP. Permit only that stdlib IPC call, not HTTP/network clients.
        caller=sys._getframe(1)
        if (caller.f_code.co_filename==socket.__file__ and
                caller.f_code.co_name=='_fallback_socketpair' and
                len(args)>1 and args[1][0] in ('127.0.0.1','::1')):
            return original_connect(*args,**kwargs)
        raise RuntimeError("Network forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
