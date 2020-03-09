#include <iostream>
#include <cstring>
#include "fasttext_api.h"

using std::cout;

// Usage: fasttext-test [train|load] train_data_path model_path
int main(int argc, char** argv) {
    auto hPtr = CreateFastText();

    if (argc < 4)
    {
        cout << "Usage: fasttext-test [train|load|nn] train_data_path model_path";
        return 1;
    }

    if (strcmp(argv[1], "train") == 0)
    {
        SupervisedArgs args;
        args.Epochs = 25;
        args.LearningRate = 1.0;
        args.WordNGrams = 3;
        args.Verbose = 2;
        args.Threads = 1;

        TrainSupervised(hPtr, argv[2], argv[3], args, nullptr);
    }
    else if (strcmp(argv[1], "load") == 0 || strcmp(argv[1], "nn") == 0)
    {
        LoadModel(hPtr, argv[3]);
    }
    else
    {
        cout << "Usage: fasttext-test [train|load|nn] train_data_path model_path";
        return 1;
    }

    if (strcmp(argv[1], "load") == 0)
    {
	    float* vectors;
	    int dim = GetSentenceVector(hPtr, "what is the difference between a new york strip and a bone-in new york cut sirloin", &vectors);
	    DestroyVector(vectors);

	    char** labels;
	    int nLabels = GetLabels(hPtr, &labels);
	    DestroyStrings(labels, nLabels);


	    char* buff;

	    float prob = PredictSingle(hPtr, "what is the difference between a new york strip and a bone-in new york cut sirloin ?", &buff);

	    char** buffers;
	    float* probs = new float[5];

	    int cnt = PredictMultiple(hPtr,"what is the difference between a new york strip and a bone-in new york cut sirloin ?", &buffers, probs, 5);

	    DestroyString(buff);
	    DestroyStrings(buffers, 5);
    }
	else
	{
		char** buffers;
	    float* probs = new float[5];

	    int cnt = GetNN(hPtr,"train", &buffers, probs, 5);

	    DestroyStrings(buffers, 5);
	}

    DestroyFastText(hPtr);
    return 0;
}
