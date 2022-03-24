# NLP_study
📚  원티드 프리온보딩 코스에서 배운 내용들을 정리합니다.

## class
> Pytorch를 사용한 모델링부터, 최신 NLP 모델들에 대해 배워요!

**Week 2-1 Pytorch Tutorial(1)** - tensor 인덱싱, 연산, 병합 및 분리 등 기본적인 tensor 조작법

**Week 2-2 Pytorch Tutorial(2)** - container, layers, loss function

**Week 2-3 Pytorch Dataloader** - Dataset, Dateloader 클래스

**Week 2-4 Pytorch Autograd** - [Autograd에 대한 유튜브 강의](https://www.youtube.com/watch?v=M0fX15_-xrY)

**Week 3-1 Word Embedding** - Sparse Word Embedding, Skip-gram, CBOW, Word2Vec, Glove, FastText

**Week 3-2 Tokenizer** - Subword Tokenizer, BPE, WordPiece, SentencePiece

**Week 3-3 Transformer, PLM** - Contextual Embedding Models, ELMo, Transformer, BERT, 논문을 읽는 법

**Week 3-4 NLP models** - GPT, ALBERT, RoBERTa

## assignment  
> 과제를 수행하며 Pytorch와 NLP모델을 학습해요!

**Week 2-1**
* 단어의 embedding 추출 및 생성
* cosine similarity 함수를 구현
* 단어들의 유사도를 cosine similarity로 비교
* 문장 embedding을 구해 문장 간 유사도 구하기

**Week 2-2**
* pandas 라이브러리를 사용한 데이터 전처리
* pre-trained BERT fine-tuning
* fine-tuning의 2가지 방법론 비교 (parameter freeze, unfreeze)

**Week 2-3**
* Custom Dataset 클래스를 구현
* dynamic padding을 만드는 함수 collate_fn을 구현
* DataLoader 클래스 사용

**Week 2-4**
* 커스텀 모듈(helper.py)에서 클래스와 함수를 임포트
* autograd의 개념 복습
* epoch, scheduler, grad_clipping
* validate(), train(), predict() 함수를 구현 및 학습
* evaluation metric 구현 (accuracy)

**Week 3-1**
* 단어 사전을 구축
* Negative Sampling 함수를 구현
* Skip-Gram 방식으로 word embedding을 학습하는 Word2Vec 클래스를 구현 및 학습

**Week 3-2**
* WordPiece Tokenzier를 학습
* 학습된 모델을 로드해 encoding과 decoding을 수행

**Week 3-3**
* [Transformer 논문](https://arxiv.org/pdf/1706.03762.pdf)을 읽고 본인 블로그에 정리

**Week 3-4**
* transformer 모델을 pytorch 라이브러리로 직접 구현 (필사)
* 텐서의 크기(shape)를 계산
* 파라미터 출력