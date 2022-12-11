#ifndef CACHE_STATE_H
#define CACHE_STATE_H
#include <vector>
#include <stdint.h>
using namespace std;

struct WayState{
    WayState() {}

    ~WayState() {}

    WayState(const WayState& src);

    int preuse = 0;
    unsigned int recency = 0;
    int num_hits = 0;
    
};

class SetState {
public:
    SetState() {};
    SetState(int assoc);
    ~SetState();
    SetState(const SetState& src);
    vector<WayState*> getWayArray() { return way_array; }
    void updateState(int way, bool is_hit, unsigned *recency_list);
    void resetState(int way);
    vector<double> flatten(bool normalize=true);

private:
    vector<WayState*> way_array;  
};

class Sample{
public:
    Sample(vector<uint64_t> _tags, uint32_t _assoc, SetState& _current_state, SetState& _next_state, uint64_t victim_id);
    ~Sample() {}
    bool updateSample (uint64_t access_tag);
    vector<double> flatten(bool normalize=true);
    bool isReady() {return ready;}

private:
    vector<uint64_t> tags;
    vector<char> reused;
    SetState *current_state;
    SetState *next_state;
    uint64_t victim;
    int reward;
    uint32_t assoc;
    bool ready;
};

class CacheState {
public:
    CacheState(uint32_t num_sets, uint32_t assoc);
    ~CacheState();
    SetState* getSetState(int set) { return this->set_array[set]; }
    vector<Sample*> getSetSampleList(int set) { return samples[set]; }
    void updateState(int set, int way, bool is_hit, unsigned int *recency_list);
    void resetState(int set, int way);
    void createNewSample(int set, int victim, SetState &current_state, SetState &next_state, 
                        vector<uint64_t> tags);
    void cleanSampleBuffer(int set);
    int totalSamples();

private:
    uint32_t numSets;
    uint32_t assoc;
    vector<SetState*> set_array;
    vector<vector<Sample*>> samples;  // one sample list per set
};








#endif // CACHE_STATE_H