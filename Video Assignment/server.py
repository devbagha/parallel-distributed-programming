import socket

port = 60000
s = socket.socket()
host = socket.gethostname()
s.bind((host, port))
s.listen(15)
print('Server listening....')

while True:
    conn, addr = s.accept()
    print('Got connection from', addr)

    data = conn.recv(1024)
    print('Server received', repr(data.decode()))

    filename = '21B-001-CS_Adeena Asif.txt'
    try:
        with open(filename, 'rb') as f:
            l = f.read(1024)
            while l:
                conn.send(l)
                print('Sent', repr(l.decode()))
                l = f.read(1024)
        print('Done sending')
        conn.send('->Thank you for connecting'.encode())
    except FileNotFoundError:
        print("File not found!")
        conn.send('->File not found!'.encode())
    
    conn.close()