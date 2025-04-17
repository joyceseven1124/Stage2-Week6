from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import torch
from gensim.models import Doc2Vec
import numpy as np
from classifier_model.MyClassifier_v10 import MyClassifier
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

class CKIPTokenizer:
    def __init__(self, model_name='albert-base'):
        # 初始化分詞器和詞性標注器
        self.word_segmenter = CkipWordSegmenter(model=model_name, device=-1)
        self.pos_tagger = CkipPosTagger(model=model_name, device=-1)
        
        # 要過濾掉的詞性
        self.stop_pos = {
            'P',  # 介系詞
            'Caa', 'Cab', 'Cba', 'Cbb',  # 各種連接詞
            'COLONCATEGORY', 'COMMACATEGORY', 'DOTCATEGORY', 'DASHCATEGORY',
            'ETCCATEGORY', 'EXCLAMATIONCATEGORY', 'PARENTHESISCATEGORY',
            'PAUSECATEGORY', 'QUESTIONCATEGORY', 'SEMICOLONCATEGORY',
            'SPCHANGECATEGORY'  # 標點符號類
        }

    def tokenize_and_tag(self, text: str):
        # 進行分詞
        segmented_text = self.word_segmenter([text])
        # 進行詞性標注
        pos_tags = self.pos_tagger(segmented_text)
        # 過濾掉特定詞性
        filtered_tokens = []
        for sentence, pos_tag in zip(segmented_text, pos_tags):
            for word, pos in zip(sentence, pos_tag):
                if pos not in self.stop_pos:
                    filtered_tokens.append(word)
        
        return filtered_tokens
    
# 加载 doc2vec 模型
doc2vec_model = Doc2Vec.load("doc2vec_model/doc2vec_model_v10")
input_dim = doc2vec_model.vector_size
classification_model = MyClassifier(input_dim=input_dim)
state_dict = torch.load("classifier_model/my_trained_model_v10.pt")
classification_model.load_state_dict(state_dict)

def predict_text(text: str) -> str:
    board_names = ['baseball', 'Boy-Girl', 'c_chat', 'hatepolitics', 'Lifeismoney', 'Military', 'pc_shopping', 'stock', 'Tech_Job']
    tokenizer = CKIPTokenizer()
    segmented_text = tokenizer.tokenize_and_tag(text)
    vector = doc2vec_model.infer_vector(segmented_text)
    vector_tensor = torch.tensor([vector], dtype=torch.float32)
    classification_model.eval()
    with torch.no_grad():
        prediction = classification_model(vector_tensor)
    predicted_class = prediction.argmax(dim=1).item()
    return board_names[predicted_class]