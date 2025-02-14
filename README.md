_For kindred APL projects, see_ [ada](https://github.com/BobMcDear/ada) _and_ [APLearn](https://github.com/BobMcDear/aplearn).

# trap

• **[Introduction](#introduction)**<br>
• **[Usage](#usage)**<br>
• **[Performance](#performance)**<br>
• **[Acknowledgements](#acknowledgements)**<br>

## Introduction

trap is an implementation of autoregressive transformers - namely, GPT2 - in [APL](https://aplwiki.com/). In addition to containing the complete definition of GPT, it also supports backpropagation and training with Adam, achieving parity with the PyTorch reference code.

Existing transformer implementations generally fall under two broad categories: A predominant fraction depend on libraries carefully crafted by experts that provide a straightforward interface to common functionalities with cutting-edge performance - PyTorch, TensorFlow, JAX, etc. While relatively easy to develop, this class of implementations involves interacting with frameworks whose underlying code tends to be quite specialized and thus difficult to understand or tweak. Truly from-scratch implementations, on the other hand, are written in low-level languages such as C or Rust, typically resorting to processor-specific vector intrinsics for optimal efficiency. They do not rely on large dependencies, but akin to the libraries behind the implementations in the first group, they can be dauntingly complex and span thousands of lines of code.

With trap, the goal is that the drawbacks of both approaches can be redressed and their advantages combined to yield a succinct self-contained implementation that is fast, simple, and portable. Though APL may strike some as a strange language of choice for deep learning, it offers benefits that are especially suitable for this field: First, the only first-class data type in APL is the multi-dimensional array, which is one of the central object of deep learning in the form of tensors. This also signifies that APL is by nature data parallel and therefore particularly amenable to parallelization. Notably, [the Co-dfns project](https://github.com/Co-dfns/Co-dfns) compiles APL code for CPUs and GPUs, exploiting the data parallel essence of APL to achieve high performance. Second, APL also almost entirely dispenses with the software-specific "noise" that bloats code in other languages, so APL code can be directly mapped to algorithms or mathematical expressions on a blackboard and vice versa, which cannot be said of the majority of programming languages. Finally, APL is extremely terse; its density might be considered a defect by some that renders APL a cryptic write-once, read-never language, but it allows for incredibly concise implementations of most algorithms. Assuming a decent grasp on APL syntax, shorter programs mean less code to maintain, debug, and understand.

## Usage

Dyalog APL is the dialect of the language used by trap, so the first step is to [install Dyalog](https://www.dyalog.com/download-zone.htm). Dyalog is free for personal uses, but a license is obligatory for commercial purposes. To compile trap, Co-dfns v5 is required; an installation guide can be found [here](https://github.com/Co-dfns/Co-dfns/blob/master/docs/INSTALL.md).

The ```TRANSFORMER``` namespace in [```APLSource/TRANSFORMER.apln```](https://github.com/BobMcDear/trap/blob/main/APLSource/TRANSFORMER.apln), which can be loaded via ```]Import # /path/to/APLSource```, exposes four main dfns:

* ```TRANSFORMER.FWD```: Performs a forward pass over the input data when called monadically, calculating output logits. Otherwise, the left argument is interpreted as target classes, and the cross-entropy loss is returned. Activation tensors are kept track of for backpropagation.
* ```TRANSFORMER.BWD```: Computes the gradients of the network's parameters. Technically, this is a non-niladic function, but its arguments are not used.
* ```TRANSFORMER.TRAIN```: Trains the transformer given an integral sequence. Mini-batches are sliced from the input sequence, so the argument to this dfn represents the entirety of the training data.
* ```TRANSFORMER.GEN```: Greedily generates tokens in an autoregressive fashion based off of an initial context.

A concrete use case of ```TRANSFORMER``` can be seen below. This snippet trains a character-level transformer on the content of the file ```input.txt```, using the characters' decimal Unicode code points as inputs to the model, and autoregressively generates 32 characters given the initial sequence ```Th```. A sample input text file is included in this repository.

```apl
]Import # /path/to/APLSource
TRANSFORMER.TRAIN ⎕UCS ⊃⎕NGET 'input.txt'
⎕UCS 64 TRANSFORMER.GEN {(1,≢⍵)⍴⍵}⎕UCS 'Th'
```

Having loaded Co-dfns, compiling ```TRANSFORMER``` can be done as follows:

```apl
transformer←'transformer' codfns.Fix ⎕SRC TRANSFORMER
```

Running the compiled version is no different from invoking the ```TRANSFORMER``` namespace:

```apl
transformer.TRAIN ⎕UCS ⊃⎕NGET 'input.txt'
⎕UCS 64 transformer.GEN {(1,≢⍵)⍴⍵}⎕UCS 'Th'
```

## Performance

Some APL features relied upon by trap are only available in Co-dfns v5, which is unfortunately substantially less efficient than v4 and orders of magnitude slower than popular scientific computing packages such as PyTorch. The good news is that the team behind Co-dfns is actively working to resolve the issues that are inhibiting it from reaching peak performance, and PyTorch-like efficiency can be expected in the near future. When the relevant Co-dfns improvements and fixes are released, this repository will be updated accordingly.

Interpreted trap is extremely slow and unusable beyond toy examples.

## Acknowledgements
Thanks to [Aaron W. Hsu](https://github.com/arcfide) for a fruitful and instructive conversation concerning the performance of Co-dfns-compiled trap.
