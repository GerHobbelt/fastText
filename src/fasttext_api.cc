#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <strstream>
#include <algorithm>
#include <cctype>
#include "fasttext.h"
#include "autotune.h"
#include "fasttext_api.h"

#define EMPTYIFNULL(a) (a == nullptr ? "" : a)

using namespace fasttext;
using std::ofstream;
using std::ios;
using std::endl;
using std::cout;

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


FT_API(int) Train(void* hPtr, const char* input, const char* output, TrainingArgs trainArgs, AutotuneArgs autotuneArgs,
        TrainProgressCallback trainCallback, AutotuneProgressCallback autotuneCallback,
        const char* label, const char* pretrainedVectors, bool debug)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    auto args = CreateArgs(trainArgs, autotuneArgs, label, pretrainedVectors);
    args.input = std::string(EMPTYIFNULL(input));
    args.output = std::string(EMPTYIFNULL(output));

    if (args.input.empty())
    {
        _lastError = "Empty input file specified!";
        return -1;
    }

    std::string outPath(EMPTYIFNULL(output));
    if (!outPath.empty() && (EndsWith(outPath, ".bin", true) || EndsWith(outPath, ".ftz", true))) {
        outPath = outPath.substr(0, outPath.length()-4);
    }

    std::string modelPath;
    std::string vectorsPath;

    if (!outPath.empty()) {
        modelPath = args.hasAutotune() && args.getAutotuneModelSize() != Args::kUnlimitedModelSize
                         ? outPath + ".ftz"
                         : outPath + ".bin";
        vectorsPath = outPath + ".vec";
    }

    if (debug)
    {
        ofstream stream("_train.txt");
        stream << "= eargs" << endl;

        stream << EMPTYIFNULL(input) << endl
            << EMPTYIFNULL(output) << endl
            << trainArgs.lr << endl
            << trainArgs.lrUpdateRate << endl
            << trainArgs.dim << endl
            << trainArgs.ws << endl
            << trainArgs.epoch << endl
            << trainArgs.minCount << endl
            << trainArgs.minCountLabel << endl
            << trainArgs.neg << endl
            << trainArgs.wordNgrams << endl
            << (int)trainArgs.loss << endl
            << (int)trainArgs.model << endl
            << trainArgs.bucket << endl
            << trainArgs.minn << endl
            << trainArgs.maxn << endl
            << trainArgs.thread << endl
            << trainArgs.t << endl
            << EMPTYIFNULL(label) << endl
            << trainArgs.verbose << endl
            << EMPTYIFNULL(pretrainedVectors) << endl
            << trainArgs.saveOutput << endl
            << trainArgs.seed << endl
            << trainArgs.qout << endl
            << trainArgs.retrain << endl
            << trainArgs.qnorm << endl
            << trainArgs.cutoff << endl
            << trainArgs.dsub << endl

            << EMPTYIFNULL(autotuneArgs.validationFile) << endl
            << EMPTYIFNULL(autotuneArgs.metric) << endl
            << autotuneArgs.predictions << endl
            << autotuneArgs.duration << endl
            << EMPTYIFNULL(autotuneArgs.modelSize) << endl
            << autotuneArgs.verbose << endl;

        stream << "= args" << endl;

        stream << args.input << endl
            << args.output << endl
            << args.lr << endl
            << args.lrUpdateRate << endl
            << args.dim << endl
            << args.ws << endl
            << args.epoch << endl
            << args.minCount << endl
            << args.minCountLabel << endl
            << args.neg << endl
            << args.wordNgrams << endl
            << (int)args.loss << endl
            << (int)args.model << endl
            << args.bucket << endl
            << args.minn << endl
            << args.maxn << endl
            << args.thread << endl
            << args.t << endl
            << args.label << endl
            << args.verbose << endl
            << args.pretrainedVectors << endl
            << args.saveOutput << endl
            << args.seed << endl
            << args.qout << endl
            << args.retrain << endl
            << args.qnorm << endl
            << args.cutoff << endl
            << args.dsub << endl
            << args.autotuneValidationFile << endl
            << args.autotuneMetric << endl
            << args.autotunePredictions << endl
            << args.autotuneDuration << endl
            << args.autotuneModelSize << endl
            << args.autotuneVerbose << endl;

        stream.close();
    }

    try {
        if (args.hasAutotune())
        {
            Autotune autotune(fastText);
            autotune.train(args, autotuneCallback);
        }
        else
            fastText->train(args, trainCallback);

        if (!modelPath.empty())
            fastText->saveModel(modelPath);

        if (!vectorsPath.empty())
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

FT_API(int) Quantize(void* hPtr, const char* output, TrainingArgs trainArgs, const char* label)
{
    if (!IsModelReady(hPtr))
    {
        _lastError = "Model is not ready!";
        return -1;
    }

    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    if (fastText->getArgs().model != model_name::sup)
    {
        _lastError = "Only supervised models can be quantized!";
        return -1;
    }

    if (output == nullptr)
    {
        _lastError = "Output is not specified!";
        return -1;
    }

    auto outPath = std::string(output);
    if (outPath.empty())
    {
        _lastError = "Output is not specified!";
        return -1;
    }

    if (EndsWith(outPath, ".bin", true))
        outPath = outPath.substr(0, outPath.length() - 4) + ".ftz";
    else if (!EndsWith(outPath, ".ftz", true))
        outPath = outPath + ".ftz";

    auto args = CreateArgs(trainArgs, AutotuneArgs(), label, nullptr);

    try {
        fastText->quantize(args);
        fastText->saveModel(outPath);

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

FT_API(int) GetWordVector(void* hPtr, const char* input, float** vector)
{
    auto fastText = static_cast<FastTextWrapper*>(hPtr);
    Vector svec(fastText->getDimension());
    std::string wordStr(input);

    try {
        fastText->getWordVector(svec, wordStr);
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

FT_API(int) Test(void* hPtr, const char* input, int k, float threshold, TestMeter** meterPtr, bool debug)
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

    if (debug)
    {
        ofstream stream("_debug.txt");
        stream << "= g" << endl << meter->nexamples_ << endl;

        stream << "= -1" << endl;
        WriteDebugMetrics(stream, meter->metrics_);

        for(auto& metrics : meter->labelMetrics_)
        {
            stream << "= " << metrics.first << endl;
            WriteDebugMetrics(stream, metrics.second);
        }

        auto curve = meter->precisionRecallCurve();
        stream << "= c" << endl << curve.size() << endl;
        for (int i = 0; i<curve.size(); ++i) {
            stream << curve[i].first << ";" << curve[i].second << endl;
        }

        stream.close();
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

void DestroyArgs(TrainingArgs* args)
{
    delete args;
}
fasttext::Args CreateArgs(TrainingArgs args, AutotuneArgs autotuneArgs, const char* label, const char* pretrainedVectors)
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

    // Autotune
    result.autotuneValidationFile = autotuneArgs.validationFile == nullptr ? "" : autotuneArgs.validationFile;
    result.autotuneDuration = autotuneArgs.duration;
    result.autotuneMetric = autotuneArgs.metric == nullptr ? "f1" : autotuneArgs.metric;
    result.autotuneModelSize = autotuneArgs.modelSize == nullptr ? "" : autotuneArgs.modelSize;
    result.autotunePredictions = autotuneArgs.predictions;
    result.autotuneVerbose = autotuneArgs.verbose;

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

void WriteDebugMetrics(ofstream& stream, Meter::Metrics& metrics)
{
    stream << metrics.gold << endl << metrics.predicted << endl << metrics.predictedGold << endl << metrics.scoreVsTrue.size() << endl;
    for (int i = 0; i<metrics.scoreVsTrue.size(); ++i) {
        stream << metrics.scoreVsTrue[i].first << ";" << metrics.scoreVsTrue[i].second << endl;
    }
}