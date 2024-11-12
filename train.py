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

epochs = 2000

tuples = []

# tuples = [(['<s>', 'a', 'a', 'b','b', 'c', 'c'],['a', 'a', 'b', 'b','c', 'c', '<e>'])]
tuples = [(['<s>', 'a', 'a', 'b','b', 'c', 'c'],['a', 'a', 'b', 'b','c', 'c', '<e>']), (['<s>', 'b', 'b', 'c','c', 'd', 'd'],['b', 'b', 'c', 'c','d', 'd', '<e>'])]
# tuples = [(['<s>', 'b', 'b', 'c','c', 'd', 'd'],['b', 'b', 'c', 'c','d', 'd', '<e>'])]

def process_data(data):
    tokenised = [tokenise(char) for char in data]
    return torch.tensor(tokenised)

data = [(process_data(input), process_data(target)) for input, target in tuples ]

for i in range(epochs):
    epoch_loss_sum = 0
    prevLoss = 100
    optim.zero_grad()
    # for input_batch,target_batch in tqdm(dataloader, total=len(dataloader)):
    for input,target in data:
        print(input, target)
        output = decoder(input)
#       # this squashes batch and sequence into one. But look up exactly how
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        epoch_loss_sum += loss.item()
        if(loss.item() > prevLoss * 4):
            print("SPIKE", input, output)
        prevLoss = loss.item()
    optim.step()
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


# I am getting these kind of results
# Related to this: https://chatgpt.com/share/67338f8a-4fa8-8007-844c-230d1872b502
# Try: batching. Increasing data set size. Shuffling
# result for aabb: babbc
# result for a: ba
# result for aa: bab
# result for aab: babb
# result for aabbc: babbcc
# result for b: bb
# result for bb: bbc
# result for bbc: bbcc
# result for bbcc: bbccd


