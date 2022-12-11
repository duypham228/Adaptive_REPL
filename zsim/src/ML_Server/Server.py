import os
import socket

# ==================================================================================== #
# CLASS SERVER
# ==================================================================================== #
class Server:
    def __init__(self, ml_model):
        self.ml_model = ml_model
        self.done_episodes     = 0
        self.complete_signal   = 0

    # start connection with client
    def start_server(self, port_no, ipaddress=''):
        # create a socket object
        s = socket.socket()
        print("Socket successfully created")

        # max wait time = 1200s
        s.settimeout(1200)

        # reserve a port on your computer in our
        # case it is 12345 but it can be anything
        port = port_no

        # next bind to the port
        # we have not typed any ip in the ip field
        # instead we have input an empty string
        # this makes the server listen to requests
        # coming from other computers on the network
        s.bind((ipaddress, port))
        print("socket binded to %s" % (port))

        # put the socket into listening mode
        s.listen(5)
        print("socket is listening")
        return s
    
    # process client request
    def process_request(self, request, connection):
        
        if len(request) == 0:
            return
        
        elif request == "make prediction":
            reply = "Request received: make inference"
            connection.send(reply.encode())
            state = []
            data = connection.recv(4096).decode()
            if len(data) == 0:
                reply = "Empty message"
            else:
                state = data.split("_")
                state = [float(numeric_str) for numeric_str in state]
            prediction = self.ml_model.predict(state)
            reply = str(prediction)
            connection.send(reply.encode())

        elif request == "new sample":
            reply = "Request received: add sample"
            connection.send(reply.encode())
            sample_data = []
            data = connection.recv(4096).decode()
            if len(data) == 0:
                reply = "Empty message"
            else:
                sample_data = data.split("_")
                sample_data = [float(numeric_str) for numeric_str in sample_data]
            self.ml_model.addSample(sample_data)
            reply = "New sample added"
            connection.send(reply.encode())

        elif request == "retrain":
            self.ml_model.retrain()
            reply = str(self.ml_model.num_retrains)
            connection.send(reply.encode())

        elif request == "disconnect":
            self.complete_signal = 1
            reply = "Server disconnected successfully"
            connection.send(reply.encode())

        else:
            reply = "unknown request"
            connection.send(reply.encode())

    def start(self, port_no, ipaddress):
        sock = self.start_server(port_no, ipaddress)
        while True:
            # Establish connection with client 0.
            conn, addr = sock.accept()
            print('Connected to ', addr)
            
            self.complete_signal = 0
            while self.complete_signal < 1:
                request = conn.recv(4096).decode()
                self.process_request(request, conn)

            # Close the connection with the client
            conn.close()
    
