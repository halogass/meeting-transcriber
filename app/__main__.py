"""python -m app [--host HOST] [--port PORT]"""

import argparse
import logging
import os

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="Meeting Transcriber")
    p.add_argument("--host", default=os.environ.get("MT_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("MT_PORT", "8765")))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("MT_PASSWORD"):
        logging.warning("Listening on %s without MT_PASSWORD: anyone on the network can open your meetings.", args.host)
    print(f"\n  Meeting Transcriber → http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}\n")
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info", ws_ping_interval=20)


if __name__ == "__main__":
    main()
