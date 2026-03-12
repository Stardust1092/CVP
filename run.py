"""
Entry point.

Desktop (HTTP):   python run.py
Mobile (HTTPS):   python run.py --https

With --https, TWO servers start simultaneously:
  HTTP  port 8000  →  desktop: http://localhost:8000   (no cert warning)
  HTTPS port 8443  →  mobile:  https://<LAN-IP>:8443   (requires cert bypass once)

Mobile cert bypass:
  Android Chrome  → tap "Advanced" → "Proceed to <IP> (unsafe)"
  iOS Safari      → tap "Show Details" → "Visit this website" → Reload
"""
import argparse
import datetime
import ipaddress
import os
import socket
import threading

import uvicorn

CERT_FILE   = "cert.pem"
KEY_FILE    = "key.pem"
HTTP_PORT   = 8000
HTTPS_PORT  = 8443


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _generate_cert(local_ip: str) -> bool:
    """Generate a self-signed TLS certificate. Returns True on success."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print(
            "[SSL] 'cryptography' not found.\n"
            "      Install: pip install cryptography"
        )
        return False

    print(f"[SSL] Generating self-signed certificate (SAN: {local_ip}) ...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CVP Local")])
    san_entries = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    try:
        san_entries.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))
    except Exception:
        pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=365)
        )
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(KEY_FILE, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(CERT_FILE, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"[SSL] Certificate saved: {CERT_FILE}, {KEY_FILE}")
    return True


def _ensure_cert(local_ip: str) -> bool:
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return True
    return _generate_cert(local_ip)


def _run_server(**kwargs):
    """Run a uvicorn server in its own thread (own event loop)."""
    uvicorn.run("app.main:app", **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--https", action="store_true",
        help="Also start HTTPS server on port 8443 (required for mobile camera)"
    )
    args = parser.parse_args()

    local_ip = _local_ip()

    # ── Always start HTTP on port 8000 (desktop) ─────────────────────────────
    print("=" * 60)
    print(f"  HTTP  → Desktop : http://localhost:{HTTP_PORT}")

    if args.https:
        if not _ensure_cert(local_ip):
            print("[SSL] Failed — HTTPS disabled, running HTTP only.")
            args.https = False
        else:
            print(f"  HTTPS → Mobile  : https://{local_ip}:{HTTPS_PORT}")
            print()
            print("  Mobile cert bypass:")
            print(f"    Chrome : Advanced → Proceed to {local_ip} (unsafe)")
            print( "    Safari : Show Details → Visit this website → Reload")

    print("=" * 60)

    common = dict(reload=False, log_level="warning")

    if args.https:
        # HTTP on 8000 in background thread
        http_thread = threading.Thread(
            target=_run_server,
            kwargs=dict(host="0.0.0.0", port=HTTP_PORT, **common),
            daemon=True,
        )
        http_thread.start()

        # HTTPS on 8443 in main thread (blocks)
        _run_server(
            host="0.0.0.0",
            port=HTTPS_PORT,
            ssl_certfile=CERT_FILE,
            ssl_keyfile=KEY_FILE,
            log_level="info",
            reload=False,
        )
    else:
        # HTTP only
        _run_server(host="0.0.0.0", port=HTTP_PORT, log_level="info", reload=False)
