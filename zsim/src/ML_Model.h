#ifndef ML_MODEL_H
#define ML_MODEL_H

#include <string>
// #include "ml_repl.h"
#include "cache_arrays.h"
#include "cache.h"
#include "repl_policies.h"
#include <stdint.h>
#include "cache_state.h"
using namespace std;

//====================================//
//          CLASS ML_MODEL            //
//====================================//
class MlModel {
public:
    MlModel(string conf_file, uint32_t _numSets, uint32_t _assoc, int buffer_size=1024);
    ~MlModel();
    
    int    predict(int set_id, vector<uint64_t> tags);
    void   retrain();
    void   sendSample(vector<double> sample);
    void   updateState(int set_id, int way_id, bool is_hit, 
                        unsigned *recency_list, uint64_t access_tag);
    string sendMessage(string msg);
    string getReply();
    void   connectServer();
    void   disconnectServer();

private:
    // connection config
    int m_port;
    int m_sock;
    int m_buffer_size;
    string m_ipaddr;
    int createSocket();

    // cache Metadata
    CacheState *cache;

    // prediction stats
    uint64_t num_samples = 0;
    uint64_t num_infer = 0;
};

#endif // ML_MODEL_H