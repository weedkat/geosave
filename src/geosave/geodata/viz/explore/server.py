"""Serve a Panel page from a script, a notebook or Google Colab."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from importlib import import_module
from typing import Any, TYPE_CHECKING
from urllib.parse import urlsplit

import panel as pn

if TYPE_CHECKING:
    from .explorer import Explorer


# Registry belongs to serving, not the displayable application state.
servers: dict[int, Explorer] = {}
STOP_TIMEOUT = 5


def serve_explorer(explorer: Explorer, *, port: int, address: str | None, **options):
    """Replace a locally owned server and serve independent explorer sessions."""
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    served = servers.get(port)
    if served is not None:
        if served is explorer:
            served.stop()
        else:
            served.close()
    check_port(port, address)
    server = start_server(
        explorer._create_session,
        port=port,
        address=address,
        iframe_height=explorer.height + 480,
        **options,
    )
    explorer.server = server
    explorer.port = port
    servers[port] = explorer
    return server


def stop_explorer(explorer: Explorer) -> None:
    """Stop and unregister an explorer's server, waiting for its port release."""
    if explorer.server is None or explorer.port is None:
        return
    explorer.server.stop()
    if isinstance(explorer.server, threading.Thread):
        explorer.server.join(timeout=STOP_TIMEOUT)
    if servers.get(explorer.port) is explorer:
        del servers[explorer.port]
    explorer.server = None


def check_port(port: int, address: str | None) -> None:
    """Raise unless `port` is free to bind on `address`.

    Raises:
        OSError: Another program listens on `port`.
    """
    # Panel binds inside its server thread, where a failure only prints.
    with socket.socket() as probe:
        # Tornado reuses addresses too, so a just-stopped server's port binds.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((address or "", port))
        except OSError as error:
            raise OSError(
                f"port {port} is in use by another program; pass another port"
            ) from error


def start_server(
    create_page: Callable[[], Any],
    *,
    title: str | None,
    port: int,
    address: str | None,
    websocket_origin: str | list[str] | None,
    threaded: bool | None,
    verbose: bool,
    open: bool,
    location: bool,
    iframe_height: int,
    **kwargs: Any,
) -> Any:
    """Serve one page per browser session, threaded in a notebook.

    Colab shows the server through its port-proxy iframe, `iframe_height`
    pixels tall.

    Returns:
        Native Panel server or background thread, with a `.stop()` method.

    Raises:
        ValueError: Colab is asked to serve unthreaded.
    """
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        shell = None
    else:
        shell = get_ipython()
    colab = type(shell).__module__.startswith("google.colab.")
    if threaded is None:
        threaded = hasattr(shell, "kernel")
    if colab and not threaded:
        raise ValueError("Colab requires threaded=True.")

    # Browser origins allowed to open the page's websocket.
    origins = [f"localhost:{port}", f"127.0.0.1:{port}"]
    if address and address not in ("localhost", "127.0.0.1", "0.0.0.0", "::"):
        origins.append(f"{address}:{port}")
    if isinstance(websocket_origin, str):
        origins.append(websocket_origin)
    elif websocket_origin:
        origins.extend(websocket_origin)
    output = None
    if colab:
        output = import_module("google.colab.output")
        proxy = output.eval_js(f"google.colab.kernel.proxyPort({port})")
        origins.append(urlsplit(proxy).netloc)

    server = pn.serve(
        create_page,
        title=title,
        port=port,
        address=address,
        websocket_origin=origins,
        threaded=threaded,
        verbose=verbose,
        show=open and not colab,
        location=location,
        **kwargs,
    )
    if output is not None and open:
        output.serve_kernel_port_as_iframe(port, height=iframe_height)
    return server
