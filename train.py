import wandb
from tqdm import tqdm
import torch
import torch.nn as nn

from decoder import Decoder, chars, tokenise

vocab_size = len(chars)
embed_dim = 32
decoder = Decoder(vocab_size,embed_dim, num_layers=4)

optim = torch.optim.Adam(decoder.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss(size_average=True)

wandb.init(project='multimodal_transformers', name='decoder')

epochs = 1000

# corpus ="aabbccaabbcc"

tuples = []
# for w1, w2, w3, w4, w5, w6 in zip(corpus, corpus[1:], corpus[2:], corpus[3:], corpus[4:], corpus[5:]):
#     input = ['<s>'] + [w1, w2, w3, w4, w5, w6]
#     target = [w1, w2, w3, w4, w5, w6, '<e>']
#     tuples.append((input, target))

tuples = [(['<s>', 'a', 'a', 'b','b', 'c', 'c'],['a', 'a', 'b', 'b','c', 'c', '<e>']), (['<s>', 'b', 'b', 'c','c', 'd', 'd'],['b', 'b', 'c', 'c','d', 'd', '<e>'])]

def process_data(data):
    tokenised = [tokenise(char) for char in data]
    return torch.tensor(tokenised)

data = [(process_data(input), process_data(target)) for input, target in tuples ]

for i in range(epochs):
    epoch_loss_sum = 0
    prevLoss = 100
    # for input_batch,target_batch in tqdm(dataloader, total=len(dataloader)):
    for input,target in data:
        print(input, target)
        optim.zero_grad()
        output = decoder(input)
#       # this squashes batch and sequence into one. But look up exactly how
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optim.step()
        if(loss.item() > prevLoss * 4):
            print("SPIKE", input, output)
        prevLoss = loss.item()
        epoch_loss_sum += loss.item()

    epoch_loss = epoch_loss_sum / len(data)
    wandb.log({'epoch loss': epoch_loss})
print(f"final loss: {epoch_loss}")
wandb.finish()

def infer(model, input_sequence):
    model.eval()
    tokens = [tokenise(char) for char in input_sequence]
    # add start token
    tokens = [tokenise('<s>')] + tokens

    with torch.no_grad():
        logits = model(torch.tensor(tokens))

    probabilities = nn.functional.softmax(logits, dim=-1)

    predicted_tokens = torch.argmax(probabilities, dim=-1)

    items = [token.item() for token in predicted_tokens]
    output_tokens = [chars[token] for token in items]
    return ''.join(output_tokens)


test_strings = ['aabb','a','aa', 'aab', 'aabbc', 'b','bb','bbc', 'bbcc']
for str in test_strings:
   result = infer(decoder, str)
   print(f"result for {str}: {result}")



