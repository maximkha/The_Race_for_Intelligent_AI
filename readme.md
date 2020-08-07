# The Race for Intelligent AI

Many companies including: OpenAI, Google/DeepMind, Microsoft, and countless others, have started the race for “truly” intelligent AI. *For the majority of this article I’ll be referencing OpenAI’s GPT series of machine learning models.* However, the question: “what does it mean to be truly intelligent?” needs to be answered. OpenAI have modelled this problem as a text transformer model. This model takes sequences of pieces of words (two character pairs) and tries to predict the next set of word parts. They claim this shows intelligence by embedding human knowledge into question and answer form. As shown in [Kevin Lacker’s article](http://lacker.io/ai/2020/07/06/giving-gpt-3-a-turing-test.html) GPT3 can factually answer many questions like:

```
Q: How many eyes does a giraffe have?
A: A giraffe has two eyes.

Q: How many legs does a frog have?
A: A frog has four legs.

Q: Are there any animals with three legs?
A: No, there are no animals with three legs.
```

However, Lacker also highlights one important *flaw* in GTP3’s answers. He discovered that if a nonsense question was asked the model would have no problem returning a nonsense answer:

```
Q: How many eyes does the sun have?
A: The sun has one eye.

Q: How many eyes does a blade of grass have?
A: A blade of grass has one eye.
```

Lacker also makes some important observations about the model when answering certain nonsense questions:

```
Q: Who was president of the United States in 1700?
A: William Penn was president of the United States in 1700.

Q: Who was president of the United States in 1650?
A: Richard Bennett was president of the United States in 1650.

Q: Who was president of the United States in 1620?
A: James I was president of the United States in 1620.

Q: Who was president of the United States in 1600?
A: Queen Elizabeth I was president of the United States in 1600.
```

Lacker says this about the answers:

> These wrong answers are actually fascinating! None of these were presidents of the United States, of course, since the US didn’t exist then. But they are all prominent political figures who were in charge of some US-related political entity around that time. In a sense, they are good guesses.

Lacker says that these answers are good guesses of the correct answer because they are US-related political entities. I suspect that GPT3 operates in a similar manner to *Word2Vec* models, in that the model develops groups e.g. people related to the US, people related to politcs, etc. in some latent representation (intermediate output of the model’s layer or layers). Unfortunately, GTP3, as of now (7-23-2020), has not been released to the public so I cannot test this hypothesis. In personal project that I am currently working on, I have developed the technology to measure this (granted, it might not scale, but that is just another issue to fix). However, to measure this, would still be a significant undertaking. The reason is that when measuring correlation between two categories like: US, and European political figures,<!--DO THIS WITH GPT2--> you need to be very careful that you’re actually measuring the actual categories and not some random correlation. Logically, for better performance, the lower layers should be disregarded because of neural networks hierarchal structure. To illustrate this lets look at a CNNs activations maximized layer by layer created by Fabio M. Graetz, the [full article, by Fabio M. Graetz with more explanations can be found here](https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030):

#### Layer 7

<img src="Pictures\FabioM.Graetz\1_UXS3jc-8z7s0hxLNAqqQRw.jpg" alt="1_UXS3jc-8z7s0hxLNAqqQRw"  /> 

#### Layer 14

<img src="Pictures\FabioM.Graetz\1_EyfWnqB6Lsrq36IcDmOs4w.jpg" alt="1_EyfWnqB6Lsrq36IcDmOs4w"  /> 

#### Layer 20

<img src="Pictures\FabioM.Graetz\1_MznCLV-oe3_HAXzbojw-zg.jpg" alt="1_MznCLV-oe3_HAXzbojw-zg"  /> 

#### Layer 30

<img src="Pictures\FabioM.Graetz\1_YT-pI4ECo68S10Zgp5x_QQ.jpg" alt="1_YT-pI4ECo68S10Zgp5x_QQ"  /> 

#### Layer 40

<img src="Pictures\FabioM.Graetz\1_HEMGatXVZLGIv4TRYFgtzQ.jpg" alt="1_HEMGatXVZLGIv4TRYFgtzQ"  /> 

The neural network was designed to classify images. We can see that its understanding of a class of image is developed by creating an hierarchal structure in which each layer constructs a feature out of layers below it, eventually leading up to the high level features such as what class an image is.

The same principle applies to any neural network. So to find these correlations it's best to start looking from top to bottom. Since the decoder’s job is to generate text/probabilities of tokens, they should be omitted from data collection. An [illustration of text transformers by Jay Alammar](http://jalammar.github.io/illustrated-transformer/):

<img src="Pictures\JayAlammar\1_eTzWiyQPKEtXF-FmIG3q-Q.png" alt="1_eTzWiyQPKEtXF-FmIG3q-Q"  /> 

However, what’s important to note is that GPT3 **is not intelligent**. GPT3 is, in a sense, the worlds most advanced search engine. It distills all of the knowledge that it has seen into its weights and when asked a question it constructs the most likely answer from the training data, which was made by scraping the internet. This is demonstrated by its responses to Lacker's questions as well as neural networks hierarchal structures. Even if the neural network performed human like computation, the **transformer architecture is impractical for human level intelligence**. The main issue with systems like GPT3 and BERT is that they are feed forward neural networks. What this means is that information can only flow from input to output. Illustration of how a feed forward neural network looks like, [illustration by Stanford](https://cs231n.github.io/neural-networks-1/):

<img src="Pictures\CS231N\neural_net2.jpg" alt="neural_net2"  />

The information flows from the input layer through both hidden layers and finally to an output layer. The key takeaway is that fundamentally BERT and GPT3 have the same structure in terms of information flow. Although attention layers in transformers can distribute information in a way that a normal neural network layer cannot, it still retains the fundamental property that it passes forward information from input to output. The first problem with feed forward neural nets is that they are inefficient. When processing information, the processing chain can often be broken down into multiple small repetitive tasks. For example, addition is a cyclical process, where single digit adders, or in a binary system full adders, can be used together to compute the final result. In a linear information system, to add three numbers there would have to be three adders chained together; this is not efficient, especially for neural networks, which would have to learn each adder unit. This is inefficient when it is possible to learn one unit and reuse it. Another issue with using feed forward neural networks to simulate "human level intelligence" is  thinking. Thinking is an optimization process. An example of this is: designing an layout for a webpage. GPT3 could do this task but it is limited in computation power. In feed forward neural networks, the amount of layers/neurons is directly correlated to its computation power. If we think as a group of layers making up one 'optimization unit' then we quickly see the neural network can only perform so many optimization steps before presenting the output. However, when discussing these operation blocks, it is important to note that such things do not usually occur in neural networks (with few exceptions due to architecture). This is because of the way backpropagation updates neural networks. Backprop, as seen in the image example, 'distills' the information hieratically. Another issue with transformers is their lack of memory. GPT2's maximum input length is 1024 tokens (two byte pairs); anything longer will not be incorporated into the model (the model uses sliding windows but it still has the 1024 byte pair limit). Humans deal with this differently; humans do not remember the exact words said, but still have an internal 'context variable' which is updated as they process new information, retaining all the information necessary to understand the conversation and help generate an answer. The gated recurrent unit (GRU) has been the best implementation of persistent memory in neural networks but GRU networks have fallen out of favor in NLP applications since the transformer's invention.

#### Next Step

The next step for AI is to create non-structured neural networks where information does not flow linearly. This would solve the current efficiency problems with such neural networks. Obviously, these neural networks would have limited applications due to the complexity of training and the numerous problems that can be solved by feed forward neural networks. However, like with any technology, there exist some problems that could be solved with such neural networks.
