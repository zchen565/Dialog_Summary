# Dialog_Summary


## file description
`.ipynb` used for training / fine-tuning 

data preparation in `data` directory (basically we can directly use my extracted test_sick.json and train_sick.json as input dialogue that I have saved for further training), the origianl process of generating is quiet messy and took long time.

Therefore, please directly use `data/test_sick.json` and `data/train_sick.json` if needed. (These two files are generated under the help of the SICK model, note the github of SICK does not provide you these two files)

The file link is : https://drive.google.com/drive/folders/1em8iSF91d4CwjYAll8uczkm-VLp7VEWe?usp=sharing

(I use paracomet as the generation model, also used "mismayil/comet-bart-ai2" as common sense knowledge generation)

the out txt is in `output` directory

## citation : code

the training process is similar to the, the whole process is same and need to preprocess the data and set parameters to different model.

https://github.com/AldoF95/bart-chat-summarizer-finetuning/blob/main/Bart_large_xsum_fine_tuned_samsum.ipynb

code citation : 

https://huggingface.co/learn/nlp-course/chapter7/5

https://github.com/AldoF95/bart-chat-summarizer-finetuning/blob/main/Bart_large_xsum_fine_tuned_samsum.ipynb

https://www.youtube.com/watch?v=CDmPBsZ09wg

https://github.com/SeungoneKim/SICK_Summarization

https://github.com/allenai/comet-atomic-2020

https://huggingface.co/mismayil/comet-bart-ai2

https://github.com/junpliu/condigsum

https://huggingface.co/spaces/evaluate-metric/bertscore

## citation : model from hugging face
facebook/bart-base

google/flan-t5-large

google/flan-t5-base

mismayil/comet-bart-ai2

## citation : dataset from hugging face
samsum

knkarthick/dialogsum