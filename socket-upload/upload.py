import json
import socket
import time

import scipy
import pandas as pd
from model.entity import Data, Vib, SenML, Payload, Header


def load_mat(path: str):
    data = []
    dic = scipy.io.loadmat(path)
    var = list(dic.keys())[3]
    con_list = [[element for element in upperElement] for upperElement in dic[var]]
    df = pd.DataFrame(con_list)
    # df = df.loc[:67108864]
    data.append(df[0].tolist())
    return data

def send_tcp_data(data: Data, address: str = 'localhost', port: int = 8401):
    json_data = json.dumps(data.to_dict())
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((address, port))
    client.send(json_data.encode())
    client.close()

address = "localhost"
external_id = "IN726100601"
external_key = "IN726100601"

port = 8401
thing_id = '4b3f21fd-c44f-4914-b66a-b727074bdd1b'
thing_key = '6b4b2a74-8bea-43a1-9a77-f5233eaa6d10'
channel_id = 'aaed840c-5b5f-456a-9099-e48ebbeca973'
tag = 'wave_acceleration_h_z'

data = load_mat(r"E:\workspace\python\py-tools\data\109.mat")

header = Header(thing_id, thing_key, channel_id, external_id, external_key)

for x in data:
    y_value = [round(val, 2) for val in x[:8192]]
    vib = Vib(y_value=y_value)
    senml = SenML(n=tag, bt=int(time.time()), vs=vib)
    payload = Payload(t=1, senml=senml)
    data = Data(header=header, payload=[payload])
    send_tcp_data(data, address, port)