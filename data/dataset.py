import json
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from datasets import load_dataset
import spacy
import re

class DialogsumDataset(Dataset):
    # 查看产出的数据

    # 参数说明

    # CUDA_VISIBLE_DEVICES="1" 
    # python3 train_summarization_context.py 
    # --finetune_weight_path="./new_weights_sick" 
    # --best_finetune_weight_path="./new_weights_sick_best" 
    # --dataset_name="samsum" 
    # --use_paracomet=True 
    # --model_name="facebook/bart-large-xsum" 
    # --relation "xIntent" 
    # --epoch=1 
    # --use_sentence_transformer True

    def __init__(self, encoder_max_len, decoder_max_len, split_type, tokenizer, extra_context=False, extra_supervision=False, paracomet=False, relation="xReason", supervision_relation="isAfter", roberta=False, sentence_transformer=False):
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.split_type = split_type
        self.tokenizer = tokenizer

        self.extra_context=extra_context
        self.extra_supervision=extra_supervision
        
        self.relation = relation
        self.paracomet= paracomet
        
        self.roberta=roberta
        self.sentence_transformer = sentence_transformer

        if (self.paracomet) and ("<" != self.relation[0]):
            self.relation = f"<|{self.relation}|>"

        self.supervision_relation = supervision_relation
        if not self.sentence_transformer:
            print(self.relation)

        else:
            if self.paracomet:
                print("PARACOMET sentence-transformer")
            else:
                print("COMET sentence-transformer")

        ##################################################

        self.data = custom_load_dataset('dialogsum', split=split_type)
        self.dialogue = self.data['dialogue']
        self.summary = self.data['summary']
        if split_type == "test":
            self.summary2 = self.data['summary2']
            self.summary3 = self.data['summary3']
        self.id = self.data['id']

        self.nlp = spacy.load('en_core_web_sm')
        
        if self.extra_context==True:
            if self.paracomet==False:
                ###########################
                # CODE FOR COMET 
                ###########################
                
                with open(f"../data/COMET_data/comet/dialogue/dialogsum/comet_{self.split_type}.json") as f:
                    self.dialogue_comet_inference = json.load(f)

                if self.roberta:
                    with open(f"../data/COMET_data/comet/dialogue/dialogsum/roberta_nli/roberta_classified_top1_{self.split_type}.json") as f:
                        self.roberta_classified_z = json.load(f)

                if self.sentence_transformer:
                    with open(f"../data/COMET_data/comet/dialogue/dialogsum/sentence_transformer/comet_{self.split_type}_z.json", "r") as f:
                        self.sentence_transformer_classified_z = json.load(f)

                
            else:
                ###########################
                # CODE FOR PARACOMET
                ###########################
                
                with open(f"../data/COMET_data/paracomet/dialogue/dialogsum/dialog_{self.split_type}_split5_collated.json") as f:
                    self.dialogue_comet_inference = json.load(f)
                
                if self.roberta:
                    with open(f"../data/COMET_data/paracomet/dialogue/dialogsum/roberta_nli/paracomet_dialogsum_roberta_classified_top1_{self.split_type}.json") as f:
                        self.roberta_classified_z = json.load(f)

                if self.sentence_transformer:
                    with open(f"../data/COMET_data/paracomet/dialogue/dialogsum/sentence_transformer/paracomet_{self.split_type}_z.json", "r") as f:
                        self.sentence_transformer_classified_z = json.load(f)

               
        
        if self.extra_supervision==True:
            if self.split_type=='train':
                if self.paracomet==False:
                    ######################
                    # CODE FOR COMET
                    ######################
                    with open(f"../data/COMET_data/comet/summary/dialogsum/comet_train_w.json") as f:
                        self.summary_comet_inference = json.load(f)
                    
                    if self.roberta:
                        with open(f"../data/COMET_data/comet/dialogue/dialogsum/roberta_nli/roberta_classified_top1_w.json")as f:
                            self.roberta_classified_w = json.load(f)

                    if sentence_transformer:
                        with open(f"../data/COMET_data/comet/summary/dialogsum/sentence_transformer/comet_train_w.json", "r") as f:
                            self.sentence_transformer_classified_w = json.load(f)

                else:
                    ########################
                    # CODE FOR PARACOMET
                    ########################
                    with open("../data/COMET_data/paracomet/summary/dialogsum/summary_train_split5_collated.json") as f:
                        self.summary_comet_inference = json.load(f)
                    
                    if self.roberta:
                        with open("../data/COMET_data/paracomet/summary/dialogsum/roberta_nli/roberta_classified_top1_w.json") as f:
                            self.roberta_classified_w = json.load(f)

                    if sentence_transformer:
                        with open("../data/COMET_data/paracomet/summary/dialogsum/sentence_transformer/paracomet_train_w.json", "r") as f:
                            self.sentence_transformer_classified_w = json.load(f)

        self.data_len = len(self.id)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # print("line 488\n")
        if self.extra_context==False:
            #(1, sequence_length)
            encoded_dialogue = self.tokenizer(self.dialogue[index], 
                                            padding='max_length', 
                                            truncation=True, 
                                            max_length=self.encoder_max_len, 
                                            return_tensors='pt')
        else:
            if self.split_type == "validation":
                dialog_id = f"dev_{self.id[index]}"

            else:
                dialog_id = f"{self.split_type}_{self.id[index]}"
            if self.sentence_transformer:
                # print("we are good !")
                cur_dialog_data = self.sentence_transformer_classified_z[dialog_id]

                dialogue = f" towards topic of {}"

                for sentence_idx in range(len(cur_dialog_data.keys())):
                    sentence = cur_dialog_data[str(sentence_idx)]["sentence"]
                    relation = cur_dialog_data[str(sentence_idx)]["relation"]
                    commonsense = cur_dialog_data[str(sentence_idx)]["out"]

                    dialogue += sentence + "\n"
                    dialogue+= '<I> '
                    dialogue+= commonsense+'.'
                    dialogue+= ' </I>'+'\n'
                # print(dialogue)
                return dialogue 
                

            elif self.paracomet==False:
                #######################
                # CODE FOR COMET
                #######################
                # extra context exist 
                # z is available
                splitted_dialogue = self.dialogue[index].replace('\r\n','\n').split('\n')
                
                def split_sentences(text, speaker):
                    doc = self.nlp(text)
                    sents = [speaker.replace(":","") + ' said "' + sent.text + '"' for sent in doc.sents]
                    return sents
                
                splitted_sentences = []
                for idx, utterance in enumerate(splitted_dialogue):
                    speaker = re.search(".*?\:",utterance)[0]
                    utterance = utterance.replace(speaker,"").strip()
                    utterance = split_sentences(utterance,speaker)
                    splitted_sentences.extend(utterance)
                    
                dialogue= ""
                idx=0
                
                for utterance in splitted_sentences:
                    dialogue+= utterance+'\n'
                    if self.split_type=='train':
                        try:
                            while True:
                                if self.dialogue_comet_inference['train_'+self.id[index]][idx]['sentence'] not in ("#Person1#:","#Person2#:"):
                                    commonsense = self.dialogue_comet_inference['train_'+self.id[index]][idx][self.relation][0].strip()
                                    # commonsense = commonsense.replace("PersonX","Person").replace("PersonY","Person")
                                    break
                                else:
                                    idx+=1
                                continue
                        except:
                            continue
                    elif self.split_type=='validation':
                        try:
                            while True:
                                if self.dialogue_comet_inference['dev_'+self.id[index]][idx]['sentence'] not in ("#Person1#:","#Person2#:"):
                                    commonsense = self.dialogue_comet_inference['dev_'+self.id[index]][idx][self.relation][0].strip()
                                    commonsense = commonsense.replace("PersonX","Person").replace("PersonY","Person")
                                    break
                                else:
                                    idx+=1
                                continue
                        except:
                            continue
                    else: # self.split_type=='test':
                        try:
                            while True:
                                if self.dialogue_comet_inference['test_'+self.id[index]][idx]['sentence'] not in ("#Person1#:","#Person2#:"):
                                    commonsense = self.dialogue_comet_inference['test_'+self.id[index]][idx][self.relation][0].strip()
                                    # commonsense = commonsense.replace("PersonX","Person").replace("PersonY","Person")
                                    break
                                else:
                                    idx+=1
                                continue

                        except:
                            continue
                    if 'none' not in commonsense:
                        dialogue+= '<I> '
                        dialogue+= commonsense+'.'
                        dialogue+= ' </I>'+'\n'
                    idx+=1
            ############################### PARACOMET START #######################################################
            else:
                if self.split_type=='validation':
                    dia = self.dialogue_comet_inference['dev'+'_'+self.id[index]]
                else:
                    dia = self.dialogue_comet_inference[self.split_type+'_'+self.id[index]]
                dialogue=""
                for _,sent in dia.items():
                    sentence = sent['sentence'].strip()
                    person = sentence.split()[0]
                    commonsense = sent[self.relation][0].strip()

                    dialogue += sentence +'\n'

                    if sentence != commonsense:
                        if ('<file_photo>' in sentence) or ('<photo_file>' in sentence) or ('<file_picture>' in sentence):
                            dialogue += "<I> " + person + " sent a photo. </I>" + '\n' 
                        elif ('<video>' in sentence) or ('<file_video>' in sentence):
                            dialogue += "<I> " + person + " sent a video. </I>" + '\n'
                        elif '<file_gif>' in sentence:
                            dialogue += "<I> " + person + " sent a file. </I>" + '\n'
                        elif ('<file_other>' in sentence) or ('<file_others>' in sentence):
                            dialogue += "<I> " + person + " sent a file. </I>" + '\n'
                        elif ('<link>' in sentence) or ('<file_link>' in sentence):
                            dialogue += "<I> " + person + " sent a link. </I>" + '\n'
                        elif '<location>' in sentence:
                            dialogue += "<I> " + person + " sent a location. </I>" + '\n'
                        else:
                            if commonsense.strip() != 'none':
                                dialogue += "<I> " + commonsense.strip() + ". </I>" + '\n'

            encoded_dialogue = self.tokenizer(dialogue,
                                            padding='max_length', 
                                            truncation=True, 
                                            max_length=self.encoder_max_len, 
                                            add_special_tokens=True,
                                            return_tensors='pt')

        # (1, sequence_length)
        #with self.tokenizer.as_target_tokenizer():
        encoded_summary = self.tokenizer(self.summary[index], 
                                            padding='max_length', 
                                            truncation=True, 
                                            max_length=self.decoder_max_len, 
                                            add_special_tokens=True,
                                            return_tensors='pt')
        
        
        model_inputs = encoded_dialogue
        model_inputs['input_ids'] = model_inputs['input_ids'].squeeze(0)
        model_inputs['attention_mask'] = model_inputs['attention_mask'].squeeze(0)
        model_inputs['labels'] = encoded_summary['input_ids']

        def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
            """
            Shift input ids one token to the right.
            """
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)

            shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
            shifted_input_ids[:, 0] = decoder_start_token_id

            if pad_token_id is None:
                raise ValueError("self.model.config.pad_token_id has to be defined.")
            # replace possible -100 values in labels by `pad_token_id`
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

            return shifted_input_ids

        #model_inputs['decoder_input_ids'] = shift_tokens_right(model_inputs['labels'].clone(),self.tokenizer.pad_token_id,0).squeeze(0)
        model_inputs['labels'] = model_inputs['labels'].squeeze(0)

        







