# %%
"""
# Prefix Fine-tuning
This notebook demonstrates an example of [prefix tuning](https://arxiv.org/pdf/2101.00190.pdf), where learned embeddings allow for a comparatively low-memory method to fine-tune behavior in a large language model via encoding behavior/information in a small number of embedding tokens. This notebook demonstrates an example of fine-tuning GPT-2 XL to translate from French to English and vice versa. The way this works is essentially as follows: we learn two prefixes, `[start_english]` and `[start_french]`. When we want to translate a French sentence to English, we would input the following prompt to the language model:

`[start_french]Bonjour! Je m'appelle Sully.[start_english]`.

We can also do the same task via few-shot prompting of the language model. At the end of this notebook, we compare (via [BLEU-score](https://en.wikipedia.org/wiki/BLEU)) the performance of prefix-tuning with few-shot translation and show a marked improvement. Ideally, we would do a comparison of fine-tuning the whole model on the dataset vs. prefix-tuning, but I don't have enough GPU memory on my machine to do that test. Anyway, we see a ~2x improvement in BLEU score via prefix-tuning versus 3-shot prompting!

## Requirements
- [Dataset](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset) (we won't use all of the dataset, only ~200k examples of the dataset for training and evaluation. If you don't have enough memory to load the whole dataset, just load a subset instead!)
- Pytorch (for machine learning)
- NTLK (for BLEU score)
- Numpy (for math)
- TQDM (for progress bars)
- HuggingFace `transformers` (for GPT-2 XL)
"""

# %%
import torch #ML
import torch.nn.functional as F
from torch.autograd import Variable #these will help us define the prefixes we will optimize
import numpy as np #math
from transformers import GPT2Tokenizer, GPT2LMHeadModel #GPT-2 XL and its tokenizer
from tqdm import tqdm #progress bar
import csv #reading the CSV
from nltk.translate.bleu_score import sentence_bleu #BLEU score computation

device = "cuda"

# %%
"""
## Load dataset
For ease of use/clarity, we'll load the English/French sentences pairs into two separate arrays that will correspond by index. This is not the best code practice, as a deletion or addition in either array will cause misalignment of the entire dataset, but for the purpose of clarity and for this notebook we'll let it slide!
"""

# %%
english_sentences = [] #array containing the English sentences as strings
french_sentences = [] #array containing the French sentences as strings

#Load the dataset
with open('/kaggle/input/hindi-english-parallel-corpus/hindi_english_parallel.csv', newline='\n', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        english_sentences.append(row[0])
        french_sentences.append(row[1])
        
        #restrict data loading to just 200k examples
        if len(english_sentences) >= 200000:
            break

# %%
"""
## Load model, tokenizer, and embeddings
In this section, we load the GPT-2 XL model and its corresponding tokenizer. Next, we look at the word embedding matrix shape to get the vocab length and embedding size. Lastly, we extract the embedding matrix from the model file, then convert it to a list of embeddings that we can index easily.
"""

# %%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl') #tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device) #model

vocab_len, embed_size = tuple(model.state_dict()['transformer.wte.weight'].shape)
embedding_matrix = model.state_dict()['transformer.wte.weight'].clone().cpu() #actual embedding matrix

#we create a custom tokenizer function here, because we want to return an array of embeddings rather than 
#an array of indices
def tokenize(string):
    """
    Tokenizes a string and converts it to a tensor of embeddings.
    
    Args:
    string (str): The input string to be tokenized.
    
    Returns:
    A tensor of embeddings for the input string.
    """
    # Tokenize the string using the tokenizer
    x = torch.tensor(tokenizer(string)['input_ids']).view(1, -1)
    
    # Compute the prompt length and embeddings
    prompt_len = x.shape[-1]
    prompt_embeddings = F.one_hot(x, num_classes=vocab_len).float() @ embedding_matrix
    
    # Return the prompt embeddings on the device
    return prompt_embeddings.to(device)

# %%
"""
## Training
Next, the fun part — learning the prefixes. First, we need to turn off the storage of gradients in the model parameters. This will save a ton of memory, as we won't have to store ~1.5B parameters worth of gradients.

Next, we define the prefixes as PyTorch autograd variables to be optimized. We also define a function that will create our example prompts along with the corresponding ground truth target.

Lastly, we define an optimizer ([Adam](https://arxiv.org/pdf/1412.6980.pdf)), as well as a linear learning rate decay, and begin training!

As a quick note, we use gradient accumulation in place of batch training because we don't have enough memory to do normal batches. There, technically, a mathematical inconsistency in how we've implemented the accumulation, but it's not super important. Since we train on many sentences of different length, but accumulate each gradient weighted equally, the resultant averaged gradient will be different than a true padded batch gradient. This isn't super important though.
"""

# %%
#Set each parameter of the model to not store gradients
for param in model.parameters():
    param.requires_grad = False

# %%
#Define learnable prompts

tuning_length = 16 #number of tokens for each learned prefix

#Define the prefixes, initializing them to a random unit norm
start_english_prompt = Variable(torch.randn(embedding_matrix.shape[-1]*tuning_length, 
                                            device=device).view(1, tuning_length, -1), requires_grad=True)
start_french_prompt = Variable(torch.randn(embedding_matrix.shape[-1]*tuning_length, 
                                           device=device).view(1, tuning_length, -1), requires_grad=True)

# %%
def create_example(lang1_str, lang2_str, start_lang1_prompt, start_lang2_prompt):
    """
    Concatenates the prefixes and the two input language strings and returns the resulting tensor and the
    target ground-truth tensor for the second language string.
    
    Args:
    lang1_str (str): The first language string.
    lang2_str (str): The second language string.
    start_lang1_prompt (torch.Tensor): The tensor representing the prefix for the first language string.
    start_lang2_prompt (torch.Tensor): The tensor representing the prefix for the second language string.
    
    Returns:
    A tuple containing the concatenated tensor of the prefixes and the two input language strings, and 
    the ground truth tensor for training.
    """
    lang1 = tokenize(lang1_str)
    lang2 = tokenize(lang2_str)
        
    out = torch.concat((start_lang1_prompt, lang1, start_lang2_prompt, lang2), dim=1)
    
    #we had an EOS token to the target so that we can know when the translation is done generating
    return out, torch.tensor(tokenizer(lang2_str + "<|endoftext|>")['input_ids']).view(1, -1)

# %%
iters = 2048 #number of iterations to run training for

optimizer = torch.optim.Adam([start_english_prompt, start_french_prompt], lr=0.1) #our optimizer
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, #our scheduler
                                              end_factor=0.001, total_iters=iters)

loss_hist = [] #keep track of our loss history

accumulation_steps = 64 #we use gradient accumulation because we don't have enough memory to do batches

pbar = tqdm(range(0, iters)) #progress bar

for step in pbar:
    optimizer.zero_grad() #zero gradients
    
    steps = 0 #keep track of how many steps we've accumulated
    total_loss = 0 #keep track okf the total loss for our records
    while steps < accumulation_steps: #accumulation loop
        index = np.random.randint(0, int(len(english_sentences)*0.8)) #pick a random sentence
        
        order = np.random.randint(0, 2) #pick a random order (i.e. en -> fr or fr-> en)
        
        #create the input and target
        if order == 0:
            embeddings, target = create_example(french_sentences[index], english_sentences[index], 
                                                start_french_prompt, start_english_prompt)
        else:
            embeddings, target = create_example(english_sentences[index], french_sentences[index],
                                                start_english_prompt, start_french_prompt)
        
        #so this is a bit annoying, but I don't have enough GPU memory to input more than a 384 length prompt 
        #during train time, so we restrict our training to examples of max length 384. This isn't a huge issue, 
        #but there may be diminished performance at test time if we are translating large paragraphs.
        if embeddings.shape[1] < 384: #memory constraints
            out = model(inputs_embeds=embeddings.to(device))['logits'] #compute logits
            loss = F.cross_entropy(out[:, -target.shape[-1]:].view(-1, out.size(-1)), #compute CE loss
                                   target.view(-1).to(device)) / accumulation_steps
            loss.backward() #backwards pass
            
            total_loss += loss.item() / accumulation_steps #keep track of loss

            steps += 1 #keep track of a successful gradient accumulation step
    
    loss_hist.append(total_loss) #keep track of loss history
    pbar.set_description(f"Loss: {total_loss}") #set the progress bar description
    
    optimizer.step() #update parameters
    scheduler.step() #update learning rate

# %%
"""
## Test the model performance
First, we define our few-shot prompts. These are hand-crafted via examples from our dataset. Next, we define some helper functions to sample from our model via these prefixes. Lastly, we compute the BLEU score for the few-shot vs. prefix tuning model.
"""

# %%
### These are our few-shot prompts for the translation task
prompt_3_shot_en_fr="""
en: Considerable attention is also given to ensuring access for aboriginal communities, particularly in the Labrador region.
fr: La proposition ne traite pas de la langue de prestation des services en particulier.

en: In 2005 Canada imported $332.4 million of crude oil from Angola.
fr: En 2005, le Canada a importé d’Angola pour 332,4 millions de dollars de pétrole brut.

en: Independent (FIT) leisure travel is expected to decrease 3 per cent, but this will be partially off-set by a 2 per cent increase in group leisure travel.
fr: Australie De l’avis des participants à l’APM, le tourisme d’agrément en provenance de l’Australie ne devrait diminuer dans l’ensemble que de 1 p.

en:
""".strip()

prompt_3_shot_fr_en="""
fr: La proposition ne traite pas de la langue de prestation des services en particulier.
en: Considerable attention is also given to ensuring access for aboriginal communities, particularly in the Labrador region.

fr: En 2005, le Canada a importé d’Angola pour 332,4 millions de dollars de pétrole brut.
en: In 2005 Canada imported $332.4 million of crude oil from Angola.

fr: Australie De l’avis des participants à l’APM, le tourisme d’agrément en provenance de l’Australie ne devrait diminuer dans l’ensemble que de 1 p.
en: Independent (FIT) leisure travel is expected to decrease 3 per cent, but this will be partially off-set by a 2 per cent increase in group leisure travel.

fr:
""".strip()

# %%
def translate_argmax(start_lang1_prompt, input_str, start_lang2_prompt, verbose=True):
    """
    Translates an input language string to the target language using the argmax method.
    
    Args:
    start_lang1_prompt (torch.Tensor): The tensor representing the prefix for the input language.
    input_str (str): The input language string to be translated.
    start_lang2_prompt (torch.Tensor): The tensor representing the prefix for the target language.
    verbose (bool): If True, the function prints the predicted tokens as they are generated. Defaults to True.
    
    Returns:
    A list of token IDs representing the translated target language string.
    """
    lang1 = tokenize(input_str) #tokenize input
    out = torch.concat((start_lang1_prompt, lang1, start_lang2_prompt), dim=1) #create input prompt
    input_len = out.shape[1] #get the input length
    
    if input_len > 512: #memory constraints :(
        return 0
    
    with torch.no_grad(): #use no_grad to save memory
        out_sequence = [] #will be our output sequence
        
        #generate our first output token
        out = model(inputs_embeds=out.to(device))
        out_sequence.append(out['logits'].argmax(dim=-1).flatten()[-1].item())
        
        if verbose:
            print(tokenizer.decode(out_sequence[-1]), end='')
        
        #we terminate generation either when we see the EOS token, *or* when the length of the generation 
        #is greater than the context length, *or* when the length of the generation is ~2x the length of 
        #the input sequence. This last termination clause is just to not waste our time when the model 
        #gets stuck on some repetitive or non-sense generation.
        while out_sequence[-1] != 50256 and len(out_sequence) + input_len < min(1024, input_len*2):
            out = model(inputs_embeds=embedding_matrix[out_sequence[-1]].view(1, -1).to(device), 
                        past_key_values=out['past_key_values']) #use KV recycling to save compute!
            out_sequence.append(out['logits'].argmax(dim=-1)[-1].item())
            if verbose:
                if out_sequence[-1] != 50256:
                    print(tokenizer.decode(out_sequence[-1]), end='')
    
    return out_sequence[:-1] #return all but the EOS token

def en_fr_3_shot_argmax(input_str, verbose=True):
    """
    Translates a given English input string to French using a 3-shot prompt.
    
    Args:
    input_str (str): The input English string to be translated.
    verbose (bool, optional): If True, the function will print the translation as it is being generated. Defaults to True.
    
    Returns:
    out_sequence (list): A list of tokens representing the generated French translation.
    """
    out = tokenize(prompt_3_shot_en_fr + " " + input_str.strip() + "\nfr:") #prepare input prompt
    input_len = out.shape[1] #get the length of the input
    
    if input_len > 512: #memory constraints :(
        return 0
    
    with torch.no_grad(): #save memory with no_grad
        out_sequence = []
        out = model(inputs_embeds=out.to(device)) #sample first token
        out_sequence.append(out['logits'].argmax(dim=-1).flatten()[-1].item())
        
        if verbose:
            print(tokenizer.decode(out_sequence[-1]), end='')
        
        #we terminate generation either when we see a newline token, *or* when the length of the generation 
        #is greater than the context length, *or* when the length of the generation is ~2x the length of 
        #the input sequence. This last termination clause is just to not waste our time when the model 
        #gets stuck on some repetitive or non-sense generation.
        while out_sequence[-1] != 198 and len(out_sequence) + input_len < min(1024, input_len*2):
            out = model(inputs_embeds=embedding_matrix[out_sequence[-1]].view(1, -1).to(device), 
                        past_key_values=out['past_key_values'])
            out_sequence.append(out['logits'].argmax(dim=-1)[-1].item())
            if verbose:
                if out_sequence[-1] != 198:
                    print(tokenizer.decode(out_sequence[-1]), end='')
    
    return out_sequence[:-1]

def fr_en_3_shot_argmax(input_str, verbose=True):
    """
    Translates a given French input string to English using a 3-shot prompt.
    
    Args:
    input_str (str): The input French string to be translated.
    verbose (bool, optional): If True, the function will print the translation as it is being generated. Defaults to True.
    
    Returns:
    out_sequence (list): A list of tokens representing the generated English translation.
    """
    out = tokenize(prompt_3_shot_fr_en + " " + input_str.strip() + "\nen:") #prepare input prompt 
    input_len = out.shape[1] #get the length of the input
    
    if input_len > 512: #memory constraints :(
        return 0
    
    with torch.no_grad(): #save memory with no_grad
        out_sequence = []
        out = model(inputs_embeds=out.to(device)) #sample first token
        out_sequence.append(out['logits'].argmax(dim=-1).flatten()[-1].item())
        
        if verbose:
            print(tokenizer.decode(out_sequence[-1]), end='')
        
        #we terminate generation either when we see a newline token, *or* when the length of the generation 
        #is greater than the context length, *or* when the length of the generation is ~2x the length of 
        #the input sequence. This last termination clause is just to not waste our time when the model 
        #gets stuck on some repetitive or non-sense generation.
        while out_sequence[-1] != 198 and len(out_sequence) + input_len < min(1024, input_len*2):
            out = model(inputs_embeds=embedding_matrix[out_sequence[-1]].view(1, -1).to(device), 
                        past_key_values=out['past_key_values'])
            out_sequence.append(out['logits'].argmax(dim=-1)[-1].item())
            if verbose:
                if out_sequence[-1] != 198:
                    print(tokenizer.decode(out_sequence[-1]), end='')
    
    return out_sequence[:-1]

# %%
### Compare BLEU scores of each approach

k_shot_BLEU = [] #will store our BLEU scores for prompting
k_shot_pairs = [] #will store our ground-truth and predicted strings

tuning_BLEU = [] #will store our BLEU scores for prefix-tuning
tuning_pairs = [] #will store our ground-truth and predicted strings


pbar = tqdm(range(int(len(english_sentences)*0.8), len(english_sentences))) #progress bar
for i in pbar:
    #get the test sentence
    en = english_sentences[i]
    fr = french_sentences[i]
    
    #translate in each direction via the 3-shot prompt
    k_shot_fr_out = tokenizer.decode(en_fr_3_shot_argmax(en, verbose=False))
    k_shot_en_out = tokenizer.decode(fr_en_3_shot_argmax(fr, verbose=False))
    
    if k_shot_fr_out == 0 or k_shot_en_out == 0: #if errors in either translation, skip it
        pass
    
    #translate in each direction via prefix-tuning
    tuning_fr_out = tokenizer.decode(translate_argmax(start_english_prompt.to(device), en, 
                                                      start_french_prompt.to(device), verbose=False))
    tuning_en_out = tokenizer.decode(translate_argmax(start_french_prompt.to(device), fr, 
                                                      start_english_prompt.to(device), verbose=False))
    if tuning_fr_out == 0 or tuning_en_out == 0: #if errors in either translation, skip it
        pass
    
    #compute BLEU scores
    k_shot_BLEU.append(sentence_bleu([en.split()], k_shot_en_out.split()))
    k_shot_BLEU.append(sentence_bleu([fr.split()], k_shot_fr_out.split()))
    
    tuning_BLEU.append(sentence_bleu([en.split()], tuning_en_out.split()))
    tuning_BLEU.append(sentence_bleu([fr.split()], tuning_fr_out.split()))
    
    #append prediction pairs
    tuning_pairs.append((en, tuning_en_out))
    tuning_pairs.append((fr, tuning_fr_out))
    
    k_shot_pairs.append((en, k_shot_en_out))
    k_shot_pairs.append((fr, k_shot_fr_out))
    
    #update progress bar
    pbar.set_description(f"3-shot BLEU: {np.mean(k_shot_BLEU):.5f}, tuning BLEU: {np.mean(tuning_BLEU):.5f}")

# %%
#display results!

print(np.mean(tuning_BLEU), np.mean(k_shot_BLEU))
print(np.std(tuning_BLEU), np.std(k_shot_BLEU))

# %%


# %%
