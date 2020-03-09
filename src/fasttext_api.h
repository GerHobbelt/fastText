#ifndef FASTTEXT_FASTTEXT_API_H
#define FASTTEXT_FASTTEXT_API_H

#include "args.h"
#include <vector>

#ifdef FASTTEXT_EXPORTS
    #if defined WIN32
        #define FT_API(RetType) extern "C" __declspec(dllexport) RetType
    #else
        #define FT_API(RetType) extern "C" RetType __attribute__((visibility("default")))
    #endif
#else
    #if defined WIN32
        #define FT_API(RetType) extern "C" __declspec(dllimport) RetType
    #else
        #define FT_API(RetType) extern "C" RetType
    #endif

#endif

#pragma pack(push, 1)
typedef struct SupervisedArgs
{
    int Epochs = 5;
    double LearningRate = 0.1;
    int WordNGrams = 1;
    int MinCharNGrams = 0;
    int MaxCharNGrams = 0;
    int Verbose = 0;
    int Threads = 0;
} SupervisedArgs;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct TrainingArgs
{
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

    bool qout;
    bool retrain;
    bool qnorm;
    size_t cutoff;
    size_t dsub;
} TrainingArgs;
#pragma pack(pop)

FT_API(void*) CreateFastText();
FT_API(void) LoadModel(void* hPtr, const char* path);
FT_API(void) LoadModelData(void* hPtr, const char* data, long length);
FT_API(void) DestroyFastText(void* hPtr);
FT_API(int) GetMaxLabelLength(void* hPtr);
FT_API(int) GetLabels(void* hPtr, char*** labels);
FT_API(int) GetNN(void* hPtr, const char* input, char*** predictedNeighbors, float* predictedProbabilities, int n);
FT_API(void) TrainSupervised(void* hPtr, const char* input, const char* output, SupervisedArgs trainArgs, const char* labelPrefix);
FT_API(void) DestroyString(char* string);
FT_API(void) DestroyStrings(char** strings, int cnt);
FT_API(float) PredictSingle(void* hPtr, const char* input, char** predicted);
FT_API(int) PredictMultiple(void* hPtr, const char* input, char*** predictedLabels, float* predictedProbabilities, int n);

FT_API(int) GetSentenceVector(void* hPtr, const char* input, float** vector);
FT_API(void) DestroyVector(float* vector);

FT_API(void) Train(void* hPtr, const char* input, const char* output, TrainingArgs trainArgs, const char* label, const char* pretrainedVectors);

fasttext::Args CreateArgs(TrainingArgs args, const char* label, const char* pretrainedVectors);

#endif //FASTTEXT_FASTTEXT_API_H
