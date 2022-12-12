# fastText Code Reading
[TOC]
## Introduction


## Best Reference

* fastText paper
* **word2vec Parameter Learning Explained**
  Many conventional way of naming (such as hidder layer/hidden state) are come from this paper. And this paper also shows the detail of parameters updating process.
* **FASTTEXT.ZIP:
COMPRESSING TEXT CLASSIFICATION MODELS**
Show some details about how to compress the model and how to pruning the token dictionary

## Tricks

* Support multi-categories classifying with each sample has one or more than one labels
* Increase model's robustness by randomly dropping word-tokens in self-surpervised training mode
* TODO: Support dynamically droping not-that-importance tokens, this can protect the scale of tokens not be to large

#### Support multi-categories classifying with each sample has one or more than one labels

fastText supports this case by an a little tricky strategy. 

multi-categories classifying using case satisfied by the combination of the strategied hidden in `FastText::supervised`, `SoftmaxLoss::forward` and `Loss::findKBest`, the detials could be found in the annotation in [code reading repository]().   

Briefly speaking, during training process, for each sample, if it has more than one target label, then although its label vector shoud in multi-hot encoding form, `FastText::supervised` will convert its label vector to one-hot encoding form by randomly setting the corresponding element (in one-hot label vector) of one of randomly choosen targe labels to 1, and setting all other elements to zero. About setting which target label's corresponding element to 1, this will be controled by the parameter `targetIndex` of `SoftmaxLoss::forward`.  

During inference process, the top-k most possible prediction results will be saved in a `heap` structure combined by `std::vector< std::pair<real, int32_t> >` and several cpp stl heap algorithms, this process is executed by `Loss::findKBest`. But not that sample, `Loss::findKBest` will filter all potentials prediction results with "score" or "weight" smaller than certain threshold, this threshold is controled by the parameter `threshold` of `Loss::findKBest`. In this way we can tagging each prediction sample at most k potential labels.
    

#### Increase model's robustness by randomly dropping word-tokens in self-surpervised training mode

In `Dictionary::discard`, if a generated random number is larger than a threshold (given by parameter `rand`), then this word-token will be drop during self-surpervised training process, this introducing of randomness can improve model's rubustness.
    
    
## Tips

Here are some tips about hard-to-understand-piece of the project, which may let reading codes be more easier. 

#### The parameters updating process is splitted in `Model::update` and `Loss::forward`

This is beacuse, the gradient calculation of the parameter matrix mapping hidden layer to output layer is depend on the loss function type, but the calculation approach of the parameters matrix mapping input tokens to hidden layer is all the same not matter which loss function you choose. So if we put the computation of hidden-to-output-layer parameters gradients into `Loss::forward` and the computation of input-to-hidden-layer parameters gradients into `Model::update`, we can get following advantages on architecture design:
* We can unify each Loss function's interface and when we need add a new loss function, we can just developing a class satisfy these interface requirements
* `Loss::forward` generates intermediate result which will be helpful to get final result of the input-to-hidden-layer parameters gradients, so we can cache these result and improve computational efficency.

#### Naming of "word" may not only refer to words, but also to labels

**In fastText, there are two kinds of tokens, word-token and label-token, they could be distincted by the fact that there is  "\_\_label\_\_" sign in the label token**.  
Actually, in fastText, the "words" sometimes mean "tokens", which includes both words or labels (with "\_\_label\_\_" sign in the token). For example `Dictionary::word2int_` also saving ids of labels, and `Dictionary::words_` also saving `entry` objects of labels, the word entry and label entry could be distincted by `entry::type`.

#### How `Dictionary::find`, `Dictionary::add`, `Dictionary::word2int_`, `Dictionary::words_` works together mapping row text tokens to token's `entry` object

Here are the functions for these relative methods and attributes: 
* `Dictionary::find`: 
Allocate each token a non-collision id with remainder and shift strategy, I call this id token-id.
* `Dictionary::add`: 
During the building of `Dictionary` object, when meeting a new token, it will update `Dictionary::word2int_` and `Dictionary::words_` based on this new token's id, make sure we can mapping the token's raw text to token's `entry` object next time without recalculation.
* `Dictionary::word2int_`:
Each element's index represents an token's id, and each element value represent that token's correponding index in token-vocab `Dictionary::words_`, with that index, we can indexing this token's detail info from `Dictionary::words_`.
* `Dictionary::words_`:
This is token-vocab, it's an `std::vector` object, and each element in it is an `entry` object which holding certain token's detail info such as token text, token type(word-token or label-token), token's char n-gram, token's appearance-count, etc. 

Each token's info will be build and put into `Dictionary::words_` during the `Dictionary` building with `Dictionary::add`. This includes several steps:
* Judging if current token is a new one, if it's a new token, executing following steps.
* Getting the token's id from token's raw text by `Dictionary::find`. 
* Building current new token's `entry` object and push it back into token-vocab `Dictionary::words_`.
* Using `Dictionary::word2int_` to record this new token's index in token vocab `Dictionary::words_`. The element value in `Dictionary::word2int_` which index equal to current token-id should be current token's correponding index in `Dictionary::words_`.

After buiding `Dictionary`, when we needs getting a raw token's detail info (for example int training/inference stage), we just needs:
* Mapping token's raw text to token-id.
* Getting token-vocab-index by indexing the element with token-id from `Dictionary::word2int_`.
* Getting token `entry` object by indexing token-vocab-index from ` Dictionary::words_`.


#### Some threads-shared variables have been wrapped by `std::atomic<>` to guarantee thread-safety

Some variables, for example, `FastText::tokenCount_` and `FastText::loss_`, is defined with wrapping by `std::atomic<>`, since this varibles could be writting and reading by all threads, so they must be thread-safe.   
`FastText::tokenCount_` is responsible for counting global word and label tokens processed by all threads, each thread will update `tokenCount_` to `tokenCount_ + 1` after one thread-local-sample has been processed by that thread, the process contains reading and writting.

## TODO

* Figure out the meaning for the gradient-normalizing technique used in `Model::update`
* Thinking about if should set a maximum value for `Model::Statr::nexamples_` in case we have a huge training data size or we will continiously training the model with incremental training data.
* If should adding a minimum learning rate value in `FastText::progressInfo`

