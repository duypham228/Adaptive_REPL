#include <vector>
#include <stdint.h>
#include "cache_state.h"
using namespace std;

// Helper functions
int argmax(vector<int> _input) {
    int max_value = _input[0];
    int max_index = 0;
    for (unsigned i=1; i < _input.size(); ++i) {
        if (_input[i] > max_value) {
            max_value = _input[i];
            max_index = i;
        }
    }
    return max_index;
}

int max(vector<int> _input) {
    int max_value = _input[0];
    for (unsigned i=1; i < _input.size(); ++i) {
        if (_input[i] > max_value)
            max_value = _input[i];
    }
    return max_value;
}

// Class Waystate
WayState::WayState(const WayState& src) {
    this->preuse = src.preuse;
    this->recency = src.recency;
    this->num_hits = src.num_hits;
}

// Class SetState
SetState::SetState(int assoc) {
    // initialize way state array
    for (int i=0; i < assoc; ++i)
        this->way_array.push_back(new WayState());
}

SetState::~SetState() {
    for (unsigned i=0; i < way_array.size(); ++i) {
        delete way_array[i];
    }
    way_array.clear();
}

SetState::SetState(const SetState& src) {
    for (unsigned i=0; i < src.way_array.size(); ++i)
        this->way_array.push_back(new WayState(*src.way_array[i]));
}

void SetState::updateState(int way, bool is_hit, unsigned *recency_list) {
    // update way preuse distance
    for (unsigned i=0; i < this->way_array.size(); ++i)
        this->way_array[way]->preuse++;

    // // update way recency
    for (unsigned i=0; i < this->way_array.size(); ++i)
        this->way_array[i]->recency = recency_list[i];

    // update way num_hits
    if (is_hit) {
        this->way_array[way]->num_hits++;
        this->way_array[way]->preuse = 0;
    }
}

void SetState::resetState(int way) {
    // reset num hits since insertion
    this->way_array[way]->num_hits = 0;
    
    // reset preuse distance
    this->way_array[way]->preuse   = 0;

    // reset recency
    this->way_array[way]->recency = 0;
}

vector<double> SetState::flatten(bool normalize) {
    vector<double> state;
    vector<int> numhit_arr;
    vector<int> preuse_arr;
    for (unsigned way=0; way < way_array.size(); ++way) {
        numhit_arr.push_back(way_array[way]->num_hits);
        preuse_arr.push_back(way_array[way]->preuse);
    }
    int max_preuse = max(preuse_arr);
    int max_numhit = max(numhit_arr);

    for (unsigned way=0; way < way_array.size(); ++way) {
        if (normalize) {
            state.push_back((double)way_array[way]->preuse/max_preuse);
            state.push_back((double)way_array[way]->recency/way_array.size());
            state.push_back((double)way_array[way]->num_hits/max_numhit);
        }
        else {
            state.push_back((double)way_array[way]->preuse/max_preuse);
            state.push_back((double)way_array[way]->recency/way_array.size());
            state.push_back((double)way_array[way]->num_hits/max_numhit);
        }
    }
    return state;
}


// Class Sample
Sample::Sample(vector<uint64_t> _tags, uint32_t _assoc, SetState& _current_state, SetState& _next_state, uint64_t victim_id){
    tags = _tags;
    current_state = new SetState(_current_state);
    for (unsigned i = 0; i < assoc; i++){
        reused.push_back('n');
    }
    next_state = new SetState(_next_state);
    assoc = _assoc;
    victim = victim_id;
    reward = 0;
    ready = false;
}

bool Sample::updateSample(uint64_t access_tag){
    unsigned int other_reused = 0;

    // Check if this is a hit
    for (unsigned int way=0; way < assoc; ++way) {
        if (access_tag == tags[way]) {
            reused[way] = 'y';
            break;
        }
    }

    for (unsigned int way=0; way < assoc; ++way) {
        if (way != victim && reused[way] == 'y')
            ++other_reused;
    }

    // Reward
    if (reused[victim] == 'n' && other_reused == assoc-1) {
        reward = 1;
        ready = true;
        return true;
    }
    else if (reused[victim] == 'y' && other_reused < assoc-1) {
        reward = -1;
        ready = true;
        return true;
    }
    else
        return false;
}

vector<double> Sample::flatten(bool normalize){
    vector<double> sample = current_state->flatten(normalize);
    vector<double> next_state_flat = next_state->flatten(normalize);
    for (unsigned i=0; i < next_state_flat.size(); ++i) {
        sample.push_back(next_state_flat[i]);
    }
    if (normalize) {
        sample.push_back((double)victim/assoc);   // also the selected action
    }
    else {
        sample.push_back(victim);
    }
    sample.push_back(reward);
    return sample;
}

// Class CacheState
CacheState::CacheState(uint32_t _numSets, uint32_t _assoc){
    numSets = _numSets;
    assoc = _assoc;

    for (unsigned int i=0; i < numSets; ++i) {
        this->set_array.push_back(new SetState(assoc));
        vector<Sample*> set_samples;
        this->samples.push_back(set_samples);
    }
}

CacheState::~CacheState() {
    for (unsigned i=0; i < set_array.size(); ++i) {
        delete set_array[i];
    }
    set_array.clear();
}

void CacheState::updateState(int set, int way, bool is_hit, unsigned int *recency_list) {
    this->set_array[set]->updateState(way, is_hit, recency_list);
}

void CacheState::resetState(int set, int way) {
    this->set_array[set]->resetState(way);
}


void CacheState::createNewSample(int set, int victim, SetState &current_state, SetState &next_state, vector<uint64_t> tags) {
    Sample *new_sample = new Sample(tags, assoc, current_state, next_state, victim);
    samples[set].push_back(new_sample);
}

void CacheState::cleanSampleBuffer(int set){
    for (unsigned i=0; i < samples[set].size(); ++i) {
        if (samples[set][i]->isReady()) {
            delete samples[set][i];
            samples[set].erase(samples[set].begin()+i);
            i=0;
        }
    }
}


int CacheState::totalSamples() {
    int size = 0;
    for (unsigned i=0; i < samples.size(); ++i) {
        size += samples[i].size();
    }
    return size;
}






