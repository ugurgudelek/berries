import base64
import paramiko
import time
client = paramiko.SSHClient()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

client.connect(hostname='193.140.108.118', username="ugudelek", password="ugudelek")

stdin, stdout, stderr = client.exec_command('ls')
for line in stdout:
    print('... ' + line.strip('\n'))
    time.sleep(1)
client.close()
