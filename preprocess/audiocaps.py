import csv
import os
import pickle

from sentence_transformers import SentenceTransformer

model_name = "sbert_mpnet"
model = SentenceTransformer("all-mpnet-base-v2")  # 768-dimensional embeddings

root_dir = "~/Audio_Datasets/audiocaps"
split = 'train'

with open(os.path.join(root_dir, 'dataset', f'{split}.csv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)[1:]

audiocap_ids, ytids, _, captions = list(map(list, zip(*lines)))
# sort captions by ytid
ytids, audiocap_ids, captions = list(zip(*sorted(zip(ytids, audiocap_ids, captions))))

text_embeds = {}
for aid, caption in zip(audiocap_ids, captions):
    text_embeds[aid] = model.encode(caption)
    print(aid, caption)

# Save text embeddings
embed_fpath = os.path.join(root_dir, f"captions_sbert_{split}.pkl")
with open(embed_fpath, "wb") as stream:
    pickle.dump(text_embeds, stream)
print("Save", embed_fpath)
