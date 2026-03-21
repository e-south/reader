from __future__ import annotations

import importlib
from typing import Final

_PATCH_SENTINEL: Final[str] = "_reader_service_worker_patch_installed"


def _build_safe_service_worker_script(file_key: str) -> str:
    output_utils = importlib.import_module("marimo._output.utils")
    notebook_id = output_utils.uri_encode_component(file_key)
    return f"""
        if ('serviceWorker' in navigator) {{
            const notebookId = '{notebook_id}';

            const postNotebookId = (worker) => {{
                if (!worker) {{
                    return;
                }}
                const send = () => worker.postMessage({{ notebookId }});
                if (worker.state === 'activated') {{
                    send();
                    return;
                }}
                worker.addEventListener(
                    'statechange',
                    () => {{
                        if (worker.state === 'activated') {{
                            send();
                        }}
                    }},
                    {{ once: true }}
                );
            }};

            const syncRegistration = (registration) => {{
                const worker =
                    registration?.active ??
                    registration?.waiting ??
                    registration?.installing;
                postNotebookId(worker);
            }};

            navigator.serviceWorker
                .register('./public-files-sw.js?v=2')
                .then((registration) => {{
                    syncRegistration(registration);
                    return registration.update().then(() => {{
                        syncRegistration(registration);
                    }});
                }})
                .catch((error) => {{
                    console.error('Error registering service worker:', error);
                }});

            navigator.serviceWorker.ready
                .then((registration) => {{
                    syncRegistration(registration);
                }})
                .catch((error) => {{
                    console.error('Error updating service worker:', error);
                }});
        }} else {{
            console.warn(
                '[marimo] Service workers are not supported at this URL. Displaying files from the /public/ directory may be disabled. ' +
                'To fix this, enable service workers by using a secure connection (https) or localhost.'
            );
        }}
        """


def install_runtime_patches() -> None:
    assets = importlib.import_module("marimo._server.api.endpoints.assets")

    if getattr(assets, _PATCH_SENTINEL, False):
        return

    def _inject_service_worker_safe(html: str, file_key: str) -> str:
        return assets.inject_script(html, _build_safe_service_worker_script(file_key))

    assets._inject_service_worker = _inject_service_worker_safe
    setattr(assets, _PATCH_SENTINEL, True)
