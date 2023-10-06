import os
import sys
import numpy as np
import torch
import pandas as pd
import pickle
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
from bsa_train import PDClassifier, BSA, predict_classes, MODEL_FN
from bsa_data_prepare import MATRIX_LEN, MATRIX_WIDTH, EMBEDDINGS_FN, AUGMENTED_EMBEDDINGS_FN, AUGMENTED_EMBEDDINGS_MAP_FN, DATA_EXTRACT_FN, IMG_NAMES_KEY, LABELS_KEY, IMAGE_NAMES_KEY, LEGACY_PREDS_KEY, ID_KEY, TRUE_LABEL_KEY

if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    DEVICE = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inferencing on", DEVICE)

    dataset = BSA.train_test_split(sys.argv[1], train_share=1)[0][0]

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=128000,
        num_workers=0,
        shuffle=False,
        collate_fn=None,
        #pin_memory=True,
        #pin_memory_device='cuda:0'
    )

    model = PDClassifier(MATRIX_LEN * MATRIX_WIDTH, len(dataset.get_classes())).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(sys.argv[1], MODEL_FN)))
    model.eval()

    # dictionary of lists
    all_predicted_labels = np.array([], dtype='uint8')
    all_img_names = []
    all_true_labels = []
    all_legacy_predictions = []
    all_predicted_scores = np.array([], dtype=np.float16)
    t_start = time.time()
    for inputs, true_labels, img_names in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs).detach()
            preds = torch.argmax(outputs, 1)
            outputs = outputs.cpu().numpy()

            predicted_class_scores = np.round(np.max(outputs, axis=1), 2)
            alt_class_scores = np.round(np.min(outputs, axis=1), 2)

        all_img_names.extend(img_names)
        all_true_labels.extend(true_labels.numpy())
        all_predicted_labels = np.concatenate((all_predicted_labels, preds))#preds.astype('uint8')
        all_predicted_scores = np.concatenate((all_predicted_scores, predicted_class_scores))

    print(f"Model time: {time.time() - t_start:.2f} seconds")

    all_true_labels = ['' if label == 3 else label for label in all_true_labels]
    df = pd.DataFrame(
        {
            ID_KEY: all_img_names,
            'predicted_score': all_predicted_scores,
            TRUE_LABEL_KEY: all_true_labels,
            'predicted_label': all_predicted_labels
        }
    )
    df.to_csv(
        os.path.join(sys.argv[1], "predictions.csv"), index=False
    )
