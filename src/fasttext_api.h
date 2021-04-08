#ifndef FASTTEXT_FASTTEXT_API_H
#define FASTTEXT_FASTTEXT_API_H

#include "args.h"
#include "fasttext.h"
#include "meter.h"
#include <vector>
#include <ostream>
#include <fstream>

using std::ofstream;

#ifdef FASTTEXT_EXPORTS
    #ifdef WIN32
        #define FT_API(RetType) extern "C" __declspec(dllexport) RetType
    #else
        #define FT_API(RetType) extern "C" RetType __attribute__((visibility("default")))
    #endif
#else
    #ifdef WIN32
        #define FT_API(RetType) extern "C" __declspec(dllimport) RetType
    #else
        #define FT_API(RetType) extern "C" RetType
    #endif

#endif

#pragma pack(push, 1)
struct TrainingArgs
{
    // Default values copied from args.cc:21
    TrainingArgs();

    // Default supervised from args.cc:110
    static TrainingArgs* DefaultSuprevised();

    double lr;
    int lrUpdateRate;
    int dim;
    int ws;
    int epoch;
    int minCount;
    int minCountLabel;
    int neg;
    int wordNgrams;
    fasttext::loss_name loss;
    fasttext::model_name model;
    int bucket;
    int minn;
    int maxn;
    int thread;
    double t;
    int verbose;
    bool saveOutput;
    int seed;

    bool qout;
    bool retrain;
    bool qnorm;
    size_t cutoff;
    size_t dsub;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct AutotuneArgs
{
    AutotuneArgs() :
        validationFile{""},
        metric{"f1"},
        predictions{1},
        duration{60 * 5},
        modelSize{""}
    {}

    const char* validationFile;
    const char* metric;
    int predictions;
    int duration;
    const char* modelSize;
    int verbose;
};
#pragma pack(pop)

using fasttext::Meter;

#pragma pack(push, 1)

struct TestMetrics
{
    explicit TestMetrics(int label, Meter::Metrics& metrics)
    {
        gold = metrics.gold;
        predicted = metrics.predicted;
        predictedGold = metrics.predictedGold;
        this->label = label;

        scoresLen = metrics.scoreVsTrue.size();

        if (scoresLen == 0)
        {
            predictedScores = nullptr;
            goldScores = nullptr;

            return;
        }

        predictedScores = new float[scoresLen];
        goldScores = new float[scoresLen];

        for (int i = 0; i<scoresLen; ++i) {
            predictedScores[i] = metrics.scoreVsTrue[i].first;
            goldScores[i] = metrics.scoreVsTrue[i].second;
        }
    }

    ~TestMetrics()
    {
        delete[] predictedScores;
        delete[] goldScores;
    }

    uint64_t gold;
    uint64_t predicted;
    uint64_t predictedGold;
    int scoresLen;
    int label;
    float* predictedScores;
    float* goldScores;
};

struct TestMeter
{
    explicit TestMeter(Meter* meter)
    {
        nexamples = meter->nexamples_;
        nlabels = meter->labelMetrics_.size();
        sourceMeter = meter;
        metrics = new TestMetrics(-1, meter->metrics_);

        if (nlabels == 0)
        {
            labelMetrics = nullptr;
            return;
        }

        labelMetrics = new TestMetrics*[nlabels];

        int cnt = -1;
        for (auto& pair : meter->labelMetrics_)
        {
            labelMetrics[++cnt] = new TestMetrics(pair.first, pair.second);
        }
    }

    ~TestMeter()
    {
        delete sourceMeter;
        delete metrics;

        for (int i = 0; i<nlabels; ++i)
        {
            if (labelMetrics[i] != nullptr)
            {
                delete labelMetrics[i];
            }
        }

        delete[] labelMetrics;
    }

    uint64_t nexamples;
    uint64_t nlabels;
    Meter* sourceMeter;
    TestMetrics* metrics;
    TestMetrics** labelMetrics;
};

#pragma pack(pop)

// Yeah, so we need to have this Public Morozov in here to circumvent some FastText design flaws.
class FastTextWrapper : public fasttext::FastText
{
public:

    bool hasArgs() {return args_ != nullptr;}
    bool hasDict() {return dict_ != nullptr;}
    bool hasModel() {return model_ != nullptr;}
};

// Progress callbacks
typedef void (* TrainProgressCallback)(float progress, float loss, double wst, double lr, int64_t eta);
typedef void (* AutotuneProgressCallback)(double progress, int32_t trials, double bestScore, double eta);

// Errors
FT_API(void) GetLastErrorText(char** error);

// Model management
FT_API(void*) CreateFastText();
FT_API(int) LoadModel(void* hPtr, const char* path);
FT_API(int) LoadModelData(void* hPtr, const char* data, long length);
FT_API(void) DestroyFastText(void* hPtr);

// Resource management
FT_API(void) DestroyString(char* string);
FT_API(void) DestroyStrings(char** strings, int cnt);
FT_API(void) DestroyVector(float* vector);

// Model info
FT_API(int) GetMaxLabelLength(void* hPtr);
FT_API(int) GetLabels(void* hPtr, char*** labels);
FT_API(bool) IsModelReady(void* hPtr);
FT_API(int) GetModelDimension(void* hPtr);

// Args
FT_API(void) GetDefaultArgs(TrainingArgs** args);
FT_API(void) GetDefaultSupervisedArgs(TrainingArgs** args);
FT_API(void) DestroyArgs(TrainingArgs* args);

// FastText commands
FT_API(int) Train(void* hPtr, const char* input, const char* output, TrainingArgs trainArgs, AutotuneArgs autotuneArgs, TrainProgressCallback trainCallback, AutotuneProgressCallback autotuneCallback, const char* label, const char* pretrainedVectors, bool debug);
FT_API(int) Quantize(void* hPtr, const char* output, TrainingArgs trainArgs, const char* label);
FT_API(int) GetNN(void* hPtr, const char* input, char*** predictedNeighbors, float* predictedProbabilities, int n);
FT_API(int) GetSentenceVector(void* hPtr, const char* input, float** vector);
FT_API(int) GetWordVector(void* hPtr, const char* input, float** vector);

// Predictions
FT_API(float) PredictSingle(void* hPtr, const char* input, char** predicted);
FT_API(int) PredictMultiple(void* hPtr, const char* input, char*** predictedLabels, float* predictedProbabilities, int n);

// Testing
FT_API(int) Test(void* hPtr, const char* input, int k, float threshold, TestMeter** meterPtr, bool debug);
FT_API(int) DestroyMeter(void* hPtr);

// Not exported
fasttext::Args CreateArgs(TrainingArgs args, AutotuneArgs autotuneArgs, const char* label, const char* pretrainedVectors);
bool EndsWith (std::string const &fullString, std::string const &ending, bool caseInsensitive = false);
void ToLowerInplace(std::string& string);
void WriteDebugMetrics(ofstream& stream, Meter::Metrics& metrics);

#endif //FASTTEXT_FASTTEXT_API_H
