import sys
from ML_Model import DQN 
from Server import Server

# ==================================================================================== #
# MAIN FUNCTION
# ==================================================================================== #
def readPortConfig(filename):
    port_num = 0
    conf_file = open(filename, "r")
    while True:
        line = conf_file.readline()
        if not line:
            break
        else:
            if line.find("port") != -1:
                port_num = int(line.split("=")[1])
                break
    return port_num

def readIPConfig(filename):
    conf_file = open(filename, "r")
    while True:
        line = conf_file.readline()
        if not line:
            break
        else:
            if line.find("ipaddress") != -1:
                return (line.split("=")[1]).rstrip().lstrip()

def main():
    app_name     = sys.argv[1]
    weight_file  = sys.argv[2]
    server_conf  = sys.argv[3]  
    num_features = int(sys.argv[4])
    num_output   = int(sys.argv[5])
    port         = readPortConfig(server_conf)
    ip_address   = readIPConfig(server_conf)

    # Currently num_features = 3 per cache line
    ml_model = DQN(num_features, num_output, 20000, app_name, _checkpt=weight_file)
    server = Server(ml_model)
    server.start(port, ip_address)
    
    
if __name__ == "__main__":
    main()