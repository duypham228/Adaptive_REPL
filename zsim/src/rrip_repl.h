#ifndef RRIP_REPL_H_
#define RRIP_REPL_H_

#include "repl_policies.h"

// Static RRIP
class SRRIPReplPolicy : public ReplPolicy {
    protected:
        // add class member variables here
        uint64_t* array;
        uint32_t numLines;
        uint32_t rpvMax;
        bool isMiss;

    public:
        // add member methods here, refer to repl_policies.h
        explicit SRRIPReplPolicy(uint32_t _numLines, uint32_t _rpvMax) :  numLines(_numLines), rpvMax(_rpvMax) {
            array = gm_calloc<uint64_t>(numLines);
            for (uint32_t i = 0; i < numLines; i++) {
              array[i] = rpvMax;
            }
        }

        ~SRRIPReplPolicy() {
            gm_free(array);
        }

        virtual void update (uint32_t id, const MemReq* req){
            // code here
            // update on Hit or Miss
            // On Miss
            if (!isMiss) {
                array[id] = 0;
            } else { // On Hit
                array[id] = 2;
                isMiss = false;
            }
        }

        virtual void replaced (uint32_t id){
            // code here
            // replaced are called only on miss case
            isMiss = true;
            array[id] = rpvMax;
        }

        template <typename C> inline uint32_t rank(const MemReq* req, C cands){
            // code here
            uint32_t bestCand = -1;
            while (true){
                bestCand = findBestCand(cands);
                if (bestCand != -1){
                    break;
                } else {
                    for (auto ci = cands.begin(); ci != cands.end(); ci.inc()) {
                        array[*ci]++;
                    }
                }
            }
            return bestCand;
        }

        DECL_RANK_BINDINGS;
    private:
        template <typename C> uint64_t findBestCand(C cands) {
            // code here
            for (auto ci = cands.begin(); ci != cands.end(); ci.inc()) {
                if (array[*ci] == rpvMax) {
                    return *ci;
                }
            }
            return -1;
        }
};
#endif // RRIP_REPL_H_
