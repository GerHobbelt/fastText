#define CATCH_CONFIG_MAIN
#include "catch.h"

#include <iostream>
#include <cstring>
#include <fstream>
#include <string>
#include <cstdio>
#include "fasttext_api.h"

using std::cout;
using std::string;

bool file_exists(const char *filename);
bool vector_has_nonzero_elements(const float* vector, int size);

TEST_CASE("Can train, load and use supervised models", "[C API]")
{
    SECTION("Can handle errors")
    {
        auto hPtr = CreateFastText();
        TrainingArgs* args;

        GetDefaultSupervisedArgs(&args);

        int result = Supervised(hPtr, "not/a/valid.file", "tests/models/test", *args, nullptr, nullptr);
        REQUIRE(result == -1);

        char* buff;
        GetLastErrorText(&buff);
        REQUIRE(std::string(buff) == "not/a/valid.file cannot be opened for training!");

        DestroyString(buff);
        DestroyArgs(args);
        DestroyFastText(hPtr);
    }

    SECTION("Can train model with super old API")
    {
        auto hPtr = CreateFastText();
        SupervisedArgs args;
        args.Epochs = 10;
        args.LearningRate = 1.0;
        args.WordNGrams = 3;
        args.Verbose = 2;
        args.Threads = 1;

        TrainSupervised(hPtr, "tests/cooking/cooking.train.txt", "tests/models/test", args, nullptr);

        DestroyFastText(hPtr);

        REQUIRE(file_exists("tests/models/test.bin"));
        REQUIRE(file_exists("tests/models/test.vec"));
    }

    SECTION("Can train model with new API")
    {
        std::remove("tests/models/test.bin");
        std::remove("tests/models/test.vec");

        REQUIRE_FALSE(file_exists("tests/models/test.bin"));
        REQUIRE_FALSE(file_exists("tests/models/test.vec"));

        auto hPtr = CreateFastText();
        TrainingArgs* args;

        GetDefaultSupervisedArgs(&args);
        Supervised(hPtr, "tests/cooking/cooking.train.txt", "tests/models/test", *args, nullptr, nullptr);

        DestroyArgs(args);
        DestroyFastText(hPtr);

        REQUIRE(file_exists("tests/models/test.bin"));
        REQUIRE(file_exists("tests/models/test.vec"));
    }

    SECTION("Can load model")
    {
        REQUIRE(file_exists("tests/models/test.bin"));
        REQUIRE(file_exists("tests/models/test.vec"));

        auto hPtr = CreateFastText();

        LoadModel(hPtr, "tests/models/test.bin");

        SECTION("Can get sentence vector")
        {
            float* vector;
            int dim = GetSentenceVector(hPtr, "what is the difference between a new york strip and a bone-in new york cut sirloin", &vector);

            REQUIRE(dim == 100);
            REQUIRE(vector_has_nonzero_elements(vector, dim));

            DestroyVector(vector);
        }

        SECTION("Can get model labels")
        {
            char** labels;
            int nLabels = GetLabels(hPtr, &labels);

            REQUIRE(nLabels == 735);

            for (int i = 0; i<nLabels; ++i) {
                REQUIRE(!string(labels[i]).empty());
            }

            DestroyStrings(labels, nLabels);
        }

        SECTION("Can predict single label")
        {
            char* buff;
	        float prob = PredictSingle(hPtr, "what is the difference between a new york strip and a bone-in new york cut sirloin ?", &buff);

	        REQUIRE(prob > 0);
	        REQUIRE(!string(buff).empty());

	        DestroyString(buff);
        }

        SECTION("Can predict multiple labels")
        {
            char** buffers;
            float* probs = new float[5];

            int cnt = PredictMultiple(hPtr,"what is the difference between a new york strip and a bone-in new york cut sirloin ?", &buffers, probs, 5);

            REQUIRE(cnt == 5);

            for (int i = 0; i<cnt; ++i) {
                REQUIRE(!string(buffers[i]).empty());
                REQUIRE(probs[i] > 0);
            }

            DestroyStrings(buffers, 5);
        }

        SECTION("Can get nearest neighbours")
        {
            char** buffers;
            float* probs = new float[5];

            int cnt = GetNN(hPtr, "train", &buffers, probs, 5);

            REQUIRE(cnt == 5);

            for (int i = 0; i<cnt; ++i) {
                REQUIRE(!string(buffers[i]).empty());
                REQUIRE(probs[i] > 0);
            }

            DestroyStrings(buffers, 5);
        }

        SECTION("Can get sentence vector")
        {
            float* vector;

            int len = GetSentenceVector(hPtr, "This is only a test!", &vector);

            REQUIRE(len == 100);
            REQUIRE(vector_has_nonzero_elements(vector, len));

            DestroyVector(vector);
        }

        DestroyFastText(hPtr);
    }
}

bool vector_has_nonzero_elements(const float* vector, int size)
{
    for (int i = 0; i<size; ++i) {
        if (vector[i] != 0)
            return true;
    }

    return false;
}

bool file_exists(const char *filename)
{
    return static_cast<bool>(std::ifstream(filename));
}
