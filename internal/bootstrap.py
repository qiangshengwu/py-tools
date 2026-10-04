import base64
import json
import os
import requests
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

def b64decode(value):
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def b64encode(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def derive(info: str, external_id: str, external_key: str):
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=external_id.encode(),
        info=info.encode(),
    ).derive(external_key.encode())


def get_bootstrap(address: str, external_id: str, external_key: str):
    # Step 1: request the challenge.
    challenge_response = requests.post(
        f"http://{address}/devices/bootstrap/challenges/{external_id}",
        timeout=10,
    )
    challenge_response.raise_for_status()
    challenge = challenge_response.json()

    # Step 2: generate the device nonce and HMAC proof.
    device_nonce = b64encode(os.urandom(32))
    proof_input = "\n".join([
        "v1",
        external_id,
        challenge["challenge_id"],
        challenge["server_nonce"],
        device_nonce,
        str(challenge["key_version"]),
    ])
    signer = hmac.HMAC(derive("magistrala-bootstrap-auth-v1", external_id, external_key), hashes.SHA256())
    signer.update(proof_input.encode())
    proof = b64encode(signer.finalize())

    # Step 3: exchange the proof for the encrypted envelope.
    config_response = requests.post(
        f"http://{address}/devices/bootstrap/configurations/{external_id}",
        json={
            "challenge_id": challenge["challenge_id"],
            "device_nonce": device_nonce,
            "proof": proof,
        },
        timeout=10,
    )
    config_response.raise_for_status()
    envelope = config_response.json()

    # Step 4: authenticate and decrypt ciphertext || GCM tag.
    aad = "\n".join([
        "bootstrap-response-v1",
        external_id,
        challenge["challenge_id"],
        challenge["server_nonce"],
        device_nonce,
        str(challenge["key_version"]),
    ])
    plaintext = AESGCM(
        derive("magistrala-bootstrap-response-v1", external_id, external_key)
    ).decrypt(
        b64decode(envelope["nonce"]),
        b64decode(envelope["ciphertext"]),
        aad.encode(),
    )
    bootstrap_response = json.loads(plaintext)
    if bootstrap_response.get("content_type") == "application/json":
        bootstrap_response["content"] = json.loads(bootstrap_response["content"])

    return bootstrap_response.values()

if __name__ == '__main__':
    address = "192.168.1.9"
    external_id = "IN726100101"
    external_key = "IN726100101"

    id, content_type, content = get_bootstrap(address, external_id, external_key)
    print(id)
    print(content_type)
    print(content)

    client_id, device_id, external_id, interval, pressure_channel, samplingPoint, samplingRate, telemetry_channel, workspace_id = content.values()
    print(client_id)
    print(device_id)
    print(external_id)
    print(interval)
    print(pressure_channel)
    print(samplingPoint)
    print(samplingRate)
    print(telemetry_channel)
    print(workspace_id)