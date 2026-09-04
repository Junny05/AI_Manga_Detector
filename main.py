import torch
import open_clip
from PIL import Image
import os
import random

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model.eval()

def embed_image(path, model, preprocess):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return model.encode_image(img)

def load_dataset(folder):
    """Reads a folder of series subfolders into a list of (series_name, image_path) pairs."""
    data = []
    for series_folder in os.listdir(folder):
        series_path = f"{folder}/{series_folder}"
        if not os.path.isdir(series_path):
            continue
        for image_file in os.listdir(series_path):
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            data.append((series_folder, f"{series_path}/{image_file}"))
    return data


class ProjectionHead(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        torch.nn.init.zeros_(self.linear.weight)  
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + self.linear(x) 


def sample_triplet(reference_data):
    series_names = list(set(name for name, _ in reference_data))
    anchor_series = random.choice(series_names)

    same_series_images = [path for name, path in reference_data if name == anchor_series]
    if len(same_series_images) < 2:
        return None

    anchor_path, positive_path = random.sample(same_series_images, 2)

    negative_series = random.choice([s for s in series_names if s != anchor_series])
    negative_candidates = [path for name, path in reference_data if name == negative_series]
    negative_path = random.choice(negative_candidates)

    return anchor_path, positive_path, negative_path


def train_projection_head(reference_data, model, preprocess, epochs=100):
    projection = ProjectionHead()
    optimizer = torch.optim.Adam(projection.parameters(), lr=0.001)
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0)

    for epoch in range(epochs):
        triplet = sample_triplet(reference_data)
        if triplet is None:
            continue
        anchor_path, positive_path, negative_path = triplet

        anchor_vec = projection(embed_image(anchor_path, model, preprocess))
        positive_vec = projection(embed_image(positive_path, model, preprocess))
        negative_vec = projection(embed_image(negative_path, model, preprocess))

        loss = triplet_loss(anchor_vec, positive_vec, negative_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    return projection


def evaluate(test_set, reference_data, model, preprocess, projection=None):
    correct = 0
    total = 0

    for true_series, test_path in test_set:
        query_vec = embed_image(test_path, model, preprocess)
        if projection is not None:
            query_vec = projection(query_vec)

        best_score = -1
        best_match = None
        for ref_series, ref_path in reference_data:
            ref_vec = embed_image(ref_path, model, preprocess)
            if projection is not None:
                ref_vec = projection(ref_vec)
            score = torch.nn.functional.cosine_similarity(query_vec, ref_vec).item()
            if score > best_score:
                best_score = score
                best_match = ref_series

        result = "correct" if best_match == true_series else f"wrong (got {best_match})"
        print(f"  {true_series}: {result}")

        if best_match == true_series:
            correct += 1
        total += 1

    return correct / total


if __name__ == "__main__":
    print("Loading data...")
    reference_data = load_dataset("references")
    test_set = load_dataset("test")
    print(f"Loaded {len(reference_data)} reference images across "
        f"{len(set(name for name, _ in reference_data))} series")
    print(f"Loaded {len(test_set)} test images")

    print("\n--- Before fine-tuning ---")
    before_acc = evaluate(test_set, reference_data, model, preprocess, projection=None)
    print(f"Top-1 accuracy: {before_acc:.1%}")

    print("\n--- Training projection head ---")
    projection = train_projection_head(reference_data, model, preprocess, epochs=100)

    print("\n--- After fine-tuning ---")
    after_acc = evaluate(test_set, reference_data, model, preprocess, projection=projection)
    print(f"Top-1 accuracy: {after_acc:.1%}")

    print(f"\nSummary: {before_acc:.1%} -> {after_acc:.1%}")