#include <string>
#include <cstring>
#include <sstream>
#include <strstream>
#include "fasttext.h"
#include "fasttext_api.h"

using namespace fasttext;

FT_API(void*) CreateFastText()
{
    auto result = new FastText();
    return result;
}

FT_API(void) LoadModel(void* hPtr, const char* path)
{
    auto fastText = static_cast<FastText*>(hPtr);
    fastText->loadModel(path);
}

FT_API(void) LoadModelData(void* hPtr, const char* data, const long length)
{
	const auto FASTTEXT_VERSION = 12; /* Version 1b */
	const auto FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;
	auto fastText = static_cast<FastText*>(hPtr);
	std::istrstream stream(data, length);
	int32_t magic;
	int32_t version;
	stream.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
	if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
		throw std::invalid_argument("Data has wrong format!");
	}
	stream.read(reinterpret_cast<char*>(&version), sizeof(int32_t));
	if (version > FASTTEXT_VERSION) {
		throw std::invalid_argument("Data has wrong format!");
	}
	fastText->loadModel(stream);
}

FT_API(void) DestroyFastText(void* hPtr)
{
    delete static_cast<FastText*>(hPtr);
}

FT_API(int) GetMaxLabelLength(void* hPtr)
{
    auto fastText = static_cast<FastText*>(hPtr);
    auto dict = fastText->getDictionary();
    int numLabels = dict->nlabels();
    int maxLen = 0;

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

FT_API(int) GetLabels(void* hPtr, char*** labels)
{
    auto fastText = static_cast<FastText*>(hPtr);
    auto dict = fastText->getDictionary();
    int numLabels = dict->nlabels();
    auto localLabels = new char*[numLabels];

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

FT_API(int) GetNN(void* hPtr, const char* input, char*** predictedNeighbors, float* predictedProbabilities, const int n)
{
    auto fastText = static_cast<FastText*>(hPtr);

    const auto predictions = fastText->getNN(input, n);
	
    if (predictions.empty())
    {
        return 0;
    }

    const int length = fmin(predictions.size(), n);
    const auto labels = new char* [length];
    for (auto i = 0; i < length; ++i)
    {
	    const auto len = predictions[i].second.length();
        labels[i] = new char[len + 1];
        predictions[i].second.copy(labels[i], len);
        labels[i][len] = '\0';
        predictedProbabilities[i] = predictions[i].first;
    }

    *(predictedNeighbors) = labels;

    return length;
}

FT_API(void) TrainSupervised(void* hPtr, const char* input, const char* output, SupervisedArgs trainArgs, const char* labelPrefix)
{
    auto fastText = static_cast<FastText*>(hPtr);
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

    auto vectorsPath = std::string(output) + ".vec";
    auto modelPath = std::string(output) + ".bin";

    fastText->train(args);
    fastText->saveModel(modelPath);
    fastText->saveVectors(vectorsPath);
}

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

FT_API(float) PredictSingle(void* hPtr, const char* input, char** predicted)
{
    auto fastText = static_cast<FastText*>(hPtr);
    std::vector<std::pair<real,std::string>> predictions;
    std::istringstream inStream(input);

    if (!fastText->predictLine(inStream, predictions, 1, 0))
    {
        return 0;
    }

    if (predictions.size() == 0)
    {
        return 0;
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
    auto fastText = static_cast<FastText*>(hPtr);
    std::vector<std::pair<real,std::string>> predictions;
    std::istringstream inStream(input);

    if (!fastText->predictLine(inStream, predictions, n, 0))
    {
        return 0;
    }

    if (predictions.size() == 0)
    {
        return 0;
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

FT_API(int) GetSentenceVector(void* hPtr, const char* input, float** vector)
{
    auto fastText = static_cast<FastText*>(hPtr);
    Vector svec(fastText->getDimension());
    std::istringstream inStream(input);

    fastText->getSentenceVector(inStream, svec);

    float* vec = new float[svec.size()];
    size_t sz = sizeof(float) * svec.size();
    memcpy(vec, svec.data(), sz);

    *vector = vec;

    return (int)svec.size();
}

FT_API(void) DestroyVector(float* vector)
{
    delete[] vector;
}

FT_API(void) Train(void* hPtr, const char* input, const char* output, TrainingArgs trainArgs, const char* label,
                   const char* pretrainedVectors)
{
    auto fastText = static_cast<FastText*>(hPtr);
    auto args = CreateArgs(trainArgs, label, pretrainedVectors);
    args.input = std::string(input);
    args.output = std::string(output);

    auto vectorsPath = std::string(output) + ".vec";
    auto modelPath = std::string(output) + ".bin";

    fastText->train(args);
    fastText->saveModel(modelPath);
    fastText->saveVectors(vectorsPath);
}

fasttext::Args CreateArgs(TrainingArgs args, const char* label, const char* pretrainedVectors)
{
    auto result = fasttext::Args();
    result.bucket = args.bucket;
    result.cutoff = args.cutoff;
    result.dim = args.dim;
    result.dsub = args.dsub;
    result.epoch = args.epoch;

    if (label != nullptr)
    {
        result.label = std::string(label);
    }

    result.loss = args.loss;
    result.lr = args.lr;
    result.lrUpdateRate = args.lrUpdateRate;
    result.maxn = args.maxn;
    result.minCount = args.minCount;
    result.minCountLabel = args.minCountLabel;
    result.minn = args.minn;
    result.model = args.model;
    result.neg = args.neg;

    if (pretrainedVectors != nullptr)
    {
        result.pretrainedVectors = std::string(pretrainedVectors);
    }

    result.qnorm = args.qnorm;
    result.qout = args.qout;
    result.retrain = args.retrain;
    result.saveOutput = args.saveOutput;
    result.t = args.t;
    result.thread = args.thread;
    result.verbose = args.verbose;
    result.wordNgrams = args.wordNgrams;
    result.ws = args.ws;

    return result;
}
