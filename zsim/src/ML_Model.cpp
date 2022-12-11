#include "ML_Model.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <string.h>
#include <fstream>

//========================================================================//
// HELPER FUNCTIONS
//========================================================================//
int readPortConfig(string filename) {
    ifstream conf_file(filename);
    string line;
    int port_num=-1;
    if (conf_file.is_open()) {
        while(getline(conf_file, line)) {
            if (line.find("port") != string::npos) {
                unsigned long split_index=0;
                for (; split_index < line.size(); ++split_index) {
                    if (line[split_index] == '=') {
                        split_index++;
                        break;
                    }
                }
                string temp;
                for (; split_index < line.size(); ++split_index) {
                    temp += line[split_index];
                }
                port_num = stoi(temp);
                break;
            }
        }
        conf_file.close();
    }
    return port_num;
}

string readIPConfig(string filename) {
    ifstream conf_file(filename);
    string line;
    string ip_address;
    if (conf_file.is_open()) {
        while(getline(conf_file, line)) {
            if (line.find("ipaddress") != string::npos) {
                unsigned long split_index=0;
                for (; split_index < line.size(); ++split_index) {
                    if (line[split_index] == '=') {
                        split_index++;
                        break;
                    }
                }
                for (; split_index < line.size(); ++split_index) {
                    if (line[split_index] != ' ')
                        ip_address += line[split_index];
                }
                break;
            }
        }
        conf_file.close();
    }
    return ip_address;
}

//========================================================================//
// CLASS ML_MODEL
//========================================================================//
MlModel::MlModel(string conf_file, uint32_t _numSets , uint32_t _assoc, int buffer_size) {
    m_port        = readPortConfig(conf_file);
    m_ipaddr      = readIPConfig(conf_file);
    m_buffer_size = buffer_size;
    m_sock        = createSocket();
    cache         = new CacheState(_numSets, _assoc);
    connectServer();
}

MlModel::~MlModel() { 
    disconnectServer();
    delete cache; 
}

int MlModel::createSocket() {
    int sock = 0;
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        throw "Socket creation error";
    return sock;
}

void  MlModel::connectServer() {
    struct sockaddr_in serv_addr;
    memset(&serv_addr, '0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(m_port);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, m_ipaddr.c_str(), &serv_addr.sin_addr) <= 0)
        throw "Invalid address / Address not supported";
    if (connect(m_sock, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
        throw "Connection Failed";
}

void MlModel::disconnectServer() { 
    string reply = sendMessage("disconnect");
    cout << reply << endl;
    close(m_sock); 
}

string MlModel::sendMessage(string msg) {
    send(m_sock, msg.c_str(), strlen(msg.c_str()), 0);
    return getReply();
}

string MlModel::getReply() {
    char buffer[m_buffer_size];
    int valread = read(m_sock, buffer, m_buffer_size);
    string reply;
    for (int i=0; i < valread; ++i) 
        reply += buffer[i];
    return reply;
}

int MlModel::predict(int set_id, vector<uint64_t> tags) {
    string reply = sendMessage("make prediction");

    // Get state vector, i.e. state of all ways in set
    vector<double> state_vector = this->cache->getSetState(set_id)->flatten(false);

    // Send state vector to the prediction server
    string data = "";
    for (unsigned i=0; i < state_vector.size(); ++i) {
        data += to_string(state_vector[i]) + "_";
    }
    data.pop_back();

    // Server replies the prediction
    int victim = stoi(sendMessage(data));

    // Save the current state of the victim set (to create new sample)
    SetState victim_set_state = *(cache->getSetState(set_id));

    // Reset the state of the victim set
    this->cache->resetState(set_id, victim);

    // Create a new sample on the simulator side
    cache->createNewSample(set_id, victim, victim_set_state, *this->cache->getSetState(set_id), tags);

    num_infer++;
    if (num_infer % 10000 == 0) {
        cout << "num inferences = " << num_infer << endl;
    }
    return victim;
}

void MlModel::updateState( int set_id, int way_id, bool is_hit, 
                                unsigned *recency_list, uint64_t access_tag) {
    this->cache->updateState(set_id, way_id, is_hit, recency_list);

    // Update training samples in the victim set 
    vector<Sample*> sample_list = this->cache->getSetSampleList(set_id);
    for (unsigned i=0; i < sample_list.size(); ++i) {
        bool ready = sample_list[i]->updateSample(access_tag);
        if (ready) {
            num_samples++;
            sendSample(sample_list[i]->flatten(false));
        }
    }

    // Remove sent samples to save memory
    this->cache->cleanSampleBuffer(set_id);

    // Periodically retrain the model every 500 sent samples
    if (num_samples > 0 && num_samples % 5000 == 0) {
        retrain();
    }
}

void MlModel::retrain() {
    string reply = sendMessage("retrain");
    long num_retrain = stoi(reply);
    if (num_retrain % 25 == 0)
        cout << "num retrain = " << num_retrain << endl;
}

void MlModel::sendSample(vector<double> sample) {
    string reply = sendMessage("new sample");

    // Send sample data to prediction server
    string data = "";
    for (unsigned i=0; i < sample.size(); ++i) {
        data += to_string(sample[i]) + "_";
    }
    data.pop_back();
    reply = sendMessage(data);
}