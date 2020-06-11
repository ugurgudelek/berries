import paramiko
from sshtunnel import SSHTunnelForwarder
import time

with SSHTunnelForwarder(
    ('193.140.108.118'),
    ssh_username='ugudelek',
    ssh_password="ugudelek",
    remote_bind_address=("10.5.150.165", 4444)
) as tunnel:

    print(tunnel.local_bind_host)
    print(tunnel.local_bind_port)

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(hostname='localhost', port=4444, username="ugur", password="2304hecilpacha*")
    print('Connection succesful!')
    # do some operations with client session
    stdin, stdout, stderr = client.exec_command("ls")
    for line in stdout:
        print('... ' + line.strip('\n'))
    stdin, stdout, stderr = client.exec_command("./script >> output.txt")
    print(stdout.channel.recv_exit_status())    # status is 0
    client.close()
print('FINISH!')
