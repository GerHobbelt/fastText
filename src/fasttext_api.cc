#include <string>
#include <cstring>
#include <sstream>
#include <strstream>
#include <algorithm>
#include <cctype>
#include "fasttext.h"
#include "fasttext_api.h"

using namespace fasttext;

TrainingArgs::TrainingArgs()
{
    lr = 0.05;
    dim = 100;
    ws = 5;
    epoch = 5;
    minCount = 5;
    minCountLabel = 0;
    neg = 5;
    wordNgrams = 1;
    loss = loss_name::ns;
    model = model_name::sg;
    bucket = 2000000;
    minn = 3;
    maxn = 6;
    thread = 12;
    lrUpdateRate = 100;
    t = 1e-4;
    verbose = 2;
    saveOutput = false;
    seed = 0;

    qout = false;
    retrain = false;
    qnorm = false;
    cutoff = 0;
    dsub = 2;
}

TrainingArgs* TrainingArgs::DefaultSuprevised()
{
    auto args = new TrainingArgs();
    args->model = model_name::sup;
    args->loss = loss_name::softmax;
    args->minCount = 1;
    args->minn = 0;
    args->maxn = 0;
    args->lr = 0.1;

    return args;
}

//---------------------------------------------------

static std::string _lastError;

FT_API(void) GetLastErrorText(char** error)
{
    auto buff = new char[_lastError.length() + 1];
    _lastError.copy(buff, _lastError.length(), 0);
    buff[_lastError.length()] = '\0';

    *error = buff;
}

//---------------------------------------------------

FT_API(void*) CreateFastText()
{
    auto result = new FastTextWrapper();
    return result;
}

FT_API(int) LoadModel(void* hPtr, const char* path)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);

    try {
        fastText->loadModel(path);

        return 0;
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }
}

FT_API(int) LoadModelData(void* hPtr, const char* data, const long length)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);

    try {
        const auto FASTTEXT_VERSION = 12; /* Version 1b */
        const auto FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

        std::istrstream stream(data, length);
        int32_t magic;
        int32_t version;
        stream.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
        if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
            throw std::invalid_argument("Model magic signature mismatch!");
        }
        stream.read(reinterpret_cast<char*>(&version), sizeof(int32_t));
        if (version > FASTTEXT_VERSION) {
            throw std::invalid_argument("Model version mismatch!");
        }
        fastText->loadModel(stream);

        return 0;
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }
}

FT_API(void) DestroyFastText(void* hPtr)
{
    delete static_cast<FastTextWrapper*>(hPtr);
}

//---------------------------------------------------

FT_API(void) DestroyString(char* string)
{
    delete[] string;
}

FT_API(void) DestroyStrings(char** strings, int cnt)
{
    for (int i = 0; i < cnt; ++i)
    {
        delete[] strings[i];
    }
    delete[] strings;
}

FT_API(void) DestroyVector(float* vector)
{
    delete[] vector;
}

//---------------------------------------------------

FT_API(int) GetMaxLabelLength(void* hPtr)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    auto dict = fastText->getDictionary();
    int numLabels = dict->nlabels();
    int maxLen = 0;

    try {
        for (int i = 0; i < numLabels; ++i)
        {
            auto label = dict->getLabel(i);
            if (label.length() > maxLen)
            {
                maxLen = label.length();
            }
        }

        return maxLen;
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }
}

FT_API(int) GetLabels(void* hPtr, char*** labels)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    auto dict = fastText->getDictionary();
    int numLabels = dict->nlabels();
    auto localLabels = new char*[numLabels] {nullptr};

    for (int i = 0; i < numLabels; ++i)
    {
        auto label = dict->getLabel(i);
        auto len = label.length();
        localLabels[i] = new char[len + 1];
        label.copy(localLabels[i], len);
        localLabels[i][len] = '\0';
    }

    *labels = localLabels;
    return numLabels;
}

FT_API(int) GetModelDimension(void* hPtr)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    if (!fastText->hasArgs())
        return 0;

    return fastText->getDimension();
}

// Due to some crazy compiler bug we need to disable optimization for this function under Windows.
#ifdef WIN32
#pragma optimize( "", off )
#endif

FT_API(bool) IsModelReady(void* hPtr)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    return fastText->hasArgs() && fastText->hasDict() && fastText->hasModel();
}

#ifdef WIN32
#pragma optimize( "", on )
#endif

FT_API(void) GetDefaultArgs(TrainingArgs** args)
{
    *args = new TrainingArgs();
}
FT_API(void) GetDefaultSupervisedArgs(TrainingArgs** args)
{
    *args = TrainingArgs::DefaultSuprevised();
}


//---------------------------------------------------

FT_API(int) Supervised(void* hPtr, const char* input, const char* output, TrainingArgs trainArgs, const char* label,
        const char* pretrainedVectors)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    auto args = CreateArgs(trainArgs, label, pretrainedVectors);
    args.input = std::string(input);
    args.output = std::string(output);
    args.model = model_name::sup;

    if (EndsWith(args.output, ".bin")) {
        args.output = args.output.substr(0, args.output.length()-4);
    }

    auto modelPath = args.output+".bin";
    auto vectorsPath = args.output+".vec";

    try {
        fastText->train(args);
        fastText->saveModel(modelPath);
        fastText->saveVectors(vectorsPath);

        return 0;
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }
}

FT_API(int) GetNN(void* hPtr, const char* input, char*** predictedNeighbors, float* predictedProbabilities, const int n)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    std::vector<std::pair<real, std::string>> predictions;

    try {
        predictions = fastText->getNN(input, n);

        if (predictions.empty()) {
            return 0;
        }
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }

    int length = fmin(predictions.size(), n);
    auto labels = new char* [length] {nullptr};
    for (auto i = 0; i<length; ++i) {
        const auto len = predictions[i].second.length();
        labels[i] = new char[len+1];
        predictions[i].second.copy(labels[i], len);
        labels[i][len] = '\0';
        predictedProbabilities[i] = predictions[i].first;
    }

    *(predictedNeighbors) = labels;

    return length;
}

FT_API(int) GetSentenceVector(void* hPtr, const char* input, float** vector)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    Vector svec(fastText->getDimension());
    std::istringstream inStream(input);

    try {
        fastText->getSentenceVector(inStream, svec);
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }

    auto vec = new float[svec.size()];
    size_t sz = sizeof(float)*svec.size();
    memcpy(vec, svec.data(), sz);

    *vector = vec;

    return (int) svec.size();
}

//---------------------------------------------------

FT_API(float) PredictSingle(void* hPtr, const char* input, char** predicted)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    std::vector<std::pair<real,std::string>> predictions;
    std::istringstream inStream(input);

    try {
        if (!fastText->predictLine(inStream, predictions, 1, 0)) {
            return 0;
        }

        if (predictions.empty()) {
            return 0;
        }
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }

    auto len = predictions[0].second.length();
    auto buff = new char[len + 1];
    predictions[0].second.copy(buff, len);
    buff[len] = '\0';

    *predicted = buff;

    return predictions[0].first;
}

FT_API(int) PredictMultiple(void* hPtr, const char* input, char*** predictedLabels, float* predictedProbabilities, int n)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    std::vector<std::pair<real,std::string>> predictions;
    std::istringstream inStream(input);

    try {
        if (!fastText->predictLine(inStream, predictions, n, 0)) {
            return 0;
        }

        if (predictions.size()==0) {
            return 0;
        }
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }

    int cnt = fmin(predictions.size(), n);
    auto labels = new char*[cnt];
    for (int i = 0; i < cnt; ++i)
    {
        auto len = predictions[i].second.length();
        labels[i] = new char[len + 1];
        predictions[i].second.copy(labels[i], len);
        labels[i][len] = '\0';
        predictedProbabilities[i] = predictions[i].first;
    }

    *(predictedLabels) = labels;

    return cnt;
}

FT_API(int) Test(void* hPtr, const char* input, int k, float threshold, TestMeter** meterPtr)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    auto meter = new Meter(false);

    std::ifstream ifs(input);
    if (!ifs.is_open()) {
        _lastError = "Unable to open test input file";
        return -1;
    }

    try {
        fastText->test(ifs, k, threshold, *meter);
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        delete meter;
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        delete meter;
        return -1;
    }

    auto testMeter = new TestMeter(meter);
    *meterPtr = testMeter;

    return 0;
}

FT_API(int) DestroyMeter(void* hPtr)
{
    auto testMeter = static_cast<TestMeter*>(hPtr);

    delete testMeter;
    return 0;
}

FT_API(int) TrainSupervised(void* hPtr, const char* input, const char* output, SupervisedArgs trainArgs, const char* labelPrefix)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    auto args = Args();
    args.verbose = trainArgs.Verbose;
    args.input = std::string(input);
    args.output = std::string(output);
    args.model = model_name::sup;
    args.loss = loss_name::softmax;
    args.minCount = 1;
    args.minn = trainArgs.MinCharNGrams;
    args.maxn = trainArgs.MaxCharNGrams;
    args.lr = trainArgs.LearningRate;
    args.wordNgrams = trainArgs.WordNGrams;
    args.epoch = trainArgs.Epochs;

    if (labelPrefix != nullptr)
    {
        args.label = std::string(labelPrefix);
    }

    if (trainArgs.Threads > 0)
    {
        args.thread = trainArgs.Threads;
    }

    try {
        auto vectorsPath = std::string(output)+".vec";
        auto modelPath = std::string(output)+".bin";

        fastText->train(args);
        fastText->saveModel(modelPath);
        fastText->saveVectors(vectorsPath);

        return 0;
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }
}

FT_API(int) Train(void* hPtr, const char* input, const char* output, TrainingArgs trainArgs, const char* label,
                   const char* pretrainedVectors)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    auto args = CreateArgs(trainArgs, label, pretrainedVectors);
    args.input = std::string(input);
    args.output = std::string(output);

    auto vectorsPath = std::string(output)+".vec";
    auto modelPath = std::string(output)+".bin";

    try {
        fastText->train(args);
        fastText->saveModel(modelPath);
        fastText->saveVectors(vectorsPath);

        return 0;
    }
    catch (std::exception& e) {
        _lastError = std::string(e.what());
        return -1;
    }
    catch (...) {
        _lastError = "Unknown error";
        return -1;
    }
}

void DestroyArgs(TrainingArgs* args)
{
    delete args;
}
fasttext::Args CreateArgs(TrainingArgs args, const char* label, const char* pretrainedVectors)
{
    auto result = fasttext::Args();

    result.lr = args.lr;
    result.lrUpdateRate = args.lrUpdateRate;
    result.dim = args.dim;
    result.ws = args.ws;
    result.epoch = args.epoch;
    result.minCount = args.minCount;
    result.minCountLabel = args.minCountLabel;
    result.neg = args.neg;
    result.wordNgrams = args.wordNgrams;
    result.loss = args.loss;
    result.model = args.model;
    result.bucket = args.bucket;
    result.minn = args.minn;
    result.maxn = args.maxn;
    result.thread = args.thread;
    result.t = args.t;

    if (label != nullptr)
    {
        result.label = std::string(label);
    }

    result.verbose = args.verbose;

    if (pretrainedVectors != nullptr)
    {
        result.pretrainedVectors = std::string(pretrainedVectors);
    }

    result.saveOutput = args.saveOutput;
    result.seed = args.seed;

    result.qout = args.qout;
    result.retrain = args.retrain;
    result.qnorm = args.qnorm;
    result.cutoff = args.cutoff;
    result.dsub = args.dsub;

    return result;
}
void ToLowerInplace(std::string& string)
{
    std::transform(string.begin(), string.end(), string.begin(), [](unsigned char c){ return std::tolower(c); });
}
bool EndsWith(const std::string& fullString, const std::string& ending, bool caseInsensitive)
{
    if (fullString.length() >= ending.length())
    {
        auto substr = fullString.substr(fullString.length() - ending.length(), ending.length());
        auto comparison = ending;

        if (caseInsensitive)
        {
            ToLowerInplace(substr);
            ToLowerInplace(comparison);
        }

        return substr == comparison;
    } else {
        return false;
    }
}
