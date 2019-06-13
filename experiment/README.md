Experiments
-----
**Performance Enhancement**
- Back-Translation

    Using a baseline model to translate a mono big corpus. then put the
    translated(MT corpus) text back to original corpus then retrain it.
    
    State : training sample + noise
    
    Result : beam search +0.4 BLEU (500w + 800w MT corpus)
- Knowledge Distiliation

  1.    Using a teacher model (BIG-transformer) generates corpus from
        original trained source corpus? (or a new mono corpus)
  2. create a small model with lower parameters then train the MT-corpus purely.
  
  State : retraing by using sample + noise corpus
  
  Result : 500w MT corpus (23.83 BLEU), 800w MT corpus (24.5 BLEU),
  Base-small: (23.65), Base (27.41)
- Competence Learning

  1.   Calculate Freq + CDF from source corpus
  2.   make sure the model is learning easy sents first then move to
       hard sents though timestep. 
       
  paper claims : x0.4 train time and +2.2 BLEU.
    
    Result the same as baseline
- Warmup Tuning

    Paper said WMT is used 10k. Marian is used 16k. Tried : 20k, 30k.
    (almost the same BLEU)
    
- AdaBound

    2019 which is better than Adam optimizer.
    
- Target space Embedding
    
    There is a 32x512 embedding is sougou's code. the space matrix +
    scaledEmbedding.

**Low Resources**
- ULR

    Qe is hard to be generated. (Using SVD) **align your
    own(fastText_multilingual)**
    
  1.   run each language's emb
  2.   emb + seed dictionary
  3.   run SVD
  4.   got a Qe
    
    Qe && Vocab is concatenate with all language together.
    
    Qk is using eng.
- Meta Learning + ULR

    There is a paper about this. (advanced work on ULR)
    
- Char level

    Sugou is using char-level (zh2en), source code shows no change in
    input layer.
    
    Notes : Deocder consumes large memory of gpu. (10Gb in beam search) 
    
    Result : **25 BLEU** (BASE : 27.41 BLEU)

- Multi-Languages

    From Google paper (GNMT). 
  1.    Padded a tag (e.g. <en>, <zh>) on each specific lang sentence.
  2.    shared the same vocab with same token
  3.    concatenate all corpus together
  4.    train!

**NMT Architecture**
- WorkFlow
  0.   **Input Layer**
  1.   x_(max_len, 256) = Text -> Word-Embedding (30000, 256)
  2.   x_(max_len, 512) = (x_ X Emb-Scale(256, 512)) + Emb-Scale-bias(1,
       512\)
  3.   x_(max_len, 512) = x_ * sqrt(512) + Pos_emb(max_len, 512) 
  5.   **Encoder Layer, 6 layers**
  -   y_ = x_
  -   x_ = MultiHeads(x_, y_, y_)
  -   x_ = Residual(x_, y_)
  -   y_ = x_
  -   x_ = FFN(x_)
  -   x_ = Residual(x_, y_)
  -   **Repeat 6 times**
  -   context_ = x_
  6.  **Decoder Input Layer**
  -  x_ = x_ X target-Emb-Scale(512, 256) + target-Emb-Scale-bias (1,
     256\)
  -  x_ = x_ X target-Word-Embedding (256, 20000)
  -  x_ = Reshape(x_, {beam, batch, max_len, vec_dim})
  7.  **Decoder Layer, 6 layers**
  -   y_ = x_
  -   x_ = MultiHeads(x_, y_, y_)
  -   x_ = Residual(x_, y_)
  -   y_ = x_
  -   x_ = MultiHeads(x_, context_, context_)
  -   x_ = Residual(x_, y_)
  -   x_ = FFN(x_)
  -   x_ = Residual(x_, y_)
  7.   **Output Layer**
  -   x_ = x_ X SoftMax_U(512, 256) + SoftMax_U_bias(1, 256)
  -   x_ = x_ X SoftMax_V(256, 20000)
  
- Component
  -   MultiHeads (Q, K, V)
  
      
  -   Residual (x_, y_)
  
      x_ = LayerNorm(x_ + y_)

Decoding
-------
- QKV concatenate computing

    When Run Multi-heads layer, first calculate these 3 matrix once.
    (512 X3, D.model)
    
    State : unable to train a well model with beam search decoding.
    (Don't know WHY, ce-mean is fine)
- input Emb with using *256 & half-precision* (Save storage)

  1.   First retrieve max_sent_len
  2.   ((max_sent_len X 256)input emb * (256 X 512)emb scale matrix) *
       sqrt(Dmodel) + (max_sent_len X 512)Pos_emb

    WordEmb contains the *46%* storage of the entire model.
    
- Loading
    
    Obvious greater performance shows while using pure Bin-reader.
    
  1.    marian is using *YAML* lib for loading model file and vocab
        file. (It is *x22* times slower than bin-reader)
  2. You should convert model into bin file before inference it.
- Device Acceleration OpenCL
   
  1. It can enable mobile with using GPU. (All chipset)
  2. only works with *ProWx* function. (only one, GEMM)
  3. all devices share with the same header (Matrix.h) but different
     implementation of each kernal function.
  - There is another simpler libarary but *Qualcom Chipset Specific*.
    (qml.h) 
  - GPU (CUDA) Support
  - CPU Support

-------
<h2 id="techinique">Experiments Notes</h2>

1. NMT architecture : transformer
2. 7 steps for preprocessing
3. BPE=32k, 4 gpus, 2500w, en-zh + zh-en
4. Performance : amun speed (200ms) // transformer speed (400ms)
5. BLEU : amun (19) // transformer (22.92) (zh-en)
6. average drop bleu for dev-set
7. ensemble is not costing much performance (can be used with different architectures but have to be the same vocab files.)
   
* QPS :
1080TI
trans : 2 processes, 2 GPUS (6.9)
amun : 2 processes, 2 GPUS (11)

<h2 id="techinique">Future Experiments Notes</h2>

* Back-Translation
* R2L Reranking
* ULR (universal lexical representation) for low-resource corpus
* Domain Adaptation
* Document level translation
* Auto-Ensemble methods
* PRE-POS-UNK replacement methods
* MPI training for large-scale
* BERT for encoding part
