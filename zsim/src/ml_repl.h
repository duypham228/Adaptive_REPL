#ifndef ML_REPL_H_
#define ML_REPL_H_

#include "repl_policies.h"
#include "ML_Model.h"
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include "hash.h"

using namespace std;

// To keep track the LRU position of cache for the reccency of ML Model
typedef struct {
    uint32_t  LRUstackposition;
} LRU_STATE;

// Machine Learning Replacement Policy
class MLReplPolicy : public ReplPolicy {
    protected:
        // add class member variables here
        MlModel *ml_model;
        uint32_t numSets;
        uint32_t assoc;
        uint32_t numLines;
        string server_conf_file;
        uint64_t num_accesses;
        bool isMiss = false;
        HashFamily* hf;

    public:
        LRU_STATE **lru;
        // add member methods here, refer to repl_policies.h
        explicit MLReplPolicy(uint32_t _numSets, uint32_t _assoc, HashFamily* _hf, string _server_conf_file){
            numSets = _numSets;
            assoc = _assoc;
            hf = _hf;
            server_conf_file = _server_conf_file;
            lru = new LRU_STATE* [numSets];
            for(uint32_t setIndex=0; setIndex < numSets; setIndex++) {
                lru[ setIndex ]  = new LRU_STATE[ assoc ];
                for(uint32_t way=0; way<assoc; way++) {
                    // initialize stack position (for true LRU)
                    lru[ setIndex ][ way ].LRUstackposition = way;
                }
            }
            ml_model = new MlModel(server_conf_file, numSets, assoc);
        }

        ~MLReplPolicy() { 
        }

        virtual void update (uint32_t id, const MemReq* req){
            int setIndex = id / assoc;
            int wayIndex = id % assoc;
            
            // Update LRU
            IncrementTimer();
            uint32_t currLRUstackposition = lru[ setIndex ][ wayIndex ].LRUstackposition;
            // Update the stack position of all lines before the current line
            // Update implies incremeting their stack positions by one
            for(uint32_t way=0; way<assoc; way++) {
                if( lru[setIndex][way].LRUstackposition < currLRUstackposition )
                    lru[setIndex][way].LRUstackposition++;
            }
            // Set the LRU stack position of new line to be zero
	        lru[ setIndex ][ wayIndex ].LRUstackposition = 0;

            uint32_t recency_list[assoc];
            for (uint32_t way=0; way<assoc; way++) {
                recency_list[way] = lru[setIndex][way].LRUstackposition;
            }
            ml_model->updateState(setIndex, wayIndex, isMiss, recency_list, id);
            if (isMiss)
                isMiss = false;

        }

        virtual void replaced (uint32_t id){
            isMiss = true;
            // Reset way state metadata are done within the prediction of rank
        }

        void IncrementTimer() { num_accesses++; }

        template <typename C> inline uint32_t rank(const MemReq* req, C cands){
            // code here
            uint32_t bestCand = -1;
            Address lineAddr = req->lineAddr;
            uint32_t setMask = numSets - 1;
            uint32_t setIndex = hf->hash(0, lineAddr) & setMask;
            vector<uint64_t> tags;
            for (auto ci = cands.begin(); ci != cands.end(); ci.inc()) {
                tags.push_back(*ci);
            }
            bestCand = ml_model->predict(setIndex, tags);
            
            return bestCand;
        }

        DECL_RANK_BINDINGS;

    private:

};
#endif // ML_REPL_H_ 