import math
import pickle
import random
import time
import numpy as np
import copy
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import Counter
from bsa_data_prepare import EMBEDDINGS_FN, AUGMENTED_EMBEDDINGS_FN, AUGMENTED_EMBEDDINGS_MAP_FN, DATA_EXTRACT_FN, LABELS_KEY, FILE_NAME_KEY, LEGACY_PREDS_KEY
MODEL_FN = "bsa_binary_classification.pth"


class PDClassifier(nn.Module):

    def __init__(self, img_features, n_classes):
        super(PDClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(img_features, img_features),
            nn.ReLU(),
            nn.Linear(img_features, img_features),
            nn.ReLU(),
            nn.Linear(img_features, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(x)


class BSA(Dataset):

    def __init__(self, embeddings, labels, names, points, legacy_preds):
        self.embeddings = embeddings
        self.labels = labels
        self.names = names
        self.classes = [0, 1]

        self.points = points
        self.legacy_preds = legacy_preds

        # we need to have a mapping from sequential number of point to its index in embeddings, labels and names
        self.seq_to_idx = np.empty(len(points), dtype=np.uint32)
        for id, p in enumerate(points):
            self.seq_to_idx[id] = p

    @staticmethod
    def augment_test(labels, names, test, all_embeddings):

        positives = []
        n_negatives = 0
        added_positives = []
        for t in test:
            if labels[t]:
                positives.append(t)
            else:
                n_negatives += 1

        n_positive_to_add = n_negatives - len(positives)

        added_labels = []
        added_names = []
        added_embeddings = np.empty((n_positive_to_add, all_embeddings.shape[1]), dtype=np.float32)
        indx = 0
        while len(added_positives) != n_positive_to_add:

            added_positives.append(positives[indx % len(positives)])
            added_labels.append(1)
            added_names.append(names[positives[indx % len(positives)]])
            added_embeddings[indx] = all_embeddings[indx % len(positives)]
            indx += 1

        return np.concatenate((labels, np.array(added_labels, dtype=np.uint8))), names + added_names, np.concatenate((test, np.array(added_positives))), np.concatenate((all_embeddings, added_embeddings))

    @staticmethod
    def augment_train(labels, names, train, test, augmented_embeddings_mapping_to_original, all_embeddings, all_aug_embeddings):

        #if not len(test):
         #   return train, all_embeddings, labels, names
        # we need to avoid augmented items originated from items belonging to test set
        exclusion_list = []
        all_augmented_ids = list(range(len(labels), len(labels) + len(all_aug_embeddings)))
        augmented_embeddings_mapping_to_original = np.array(augmented_embeddings_mapping_to_original)
        for index in test:
            if index in augmented_embeddings_mapping_to_original:
                exclusion_list.extend(len(labels) + (np.where(augmented_embeddings_mapping_to_original == index)[0]))

        filtered_augmented_ids = []
        for i in all_augmented_ids:
            if i not in exclusion_list:
                filtered_augmented_ids.append(i)

        all_embeddings = np.concatenate((all_embeddings, all_aug_embeddings), dtype=np.float32)
        all_labels = np.concatenate((labels, np.array([1 for i in range(len(all_aug_embeddings))], dtype=np.uint8)))
        all_names = np.concatenate((names, ['X' for i in range(len(all_aug_embeddings))]))

        train = np.concatenate((train, np.array(filtered_augmented_ids, dtype=np.int32)), dtype=np.int32)
        return train, all_embeddings, all_labels, all_names

    @staticmethod
    def train_test_split(root_dir, train_share, n_splits=1):

        # detect if images have embeddings
        try:
            with open(os.path.join(root_dir, EMBEDDINGS_FN), 'rb') as f:
                embeddings = pickle.load(f)

            with open(os.path.join(root_dir, AUGMENTED_EMBEDDINGS_FN), 'rb') as f:
                all_aug_embeddings = pickle.load(f)

            with open(os.path.join(root_dir, AUGMENTED_EMBEDDINGS_MAP_FN), 'rb') as f:
                augmented_embeddings_mapping_to_original = pickle.load(f)

            with open(os.path.join(root_dir, DATA_EXTRACT_FN), 'rb') as f:
                data = pickle.load(f)

            labels = data[LABELS_KEY]
            names = data[FILE_NAME_KEY]
            legacy_preds = data[LEGACY_PREDS_KEY]

        except FileNotFoundError:
            # first you need to generate "embeddings.pkl"
            assert False

        # generate split
        ids = list(range(len(labels)))
        test_splits, train_splits = [], []
        datasets = []
        if train_share >= 1:
            # test, train = [], ids
            test_splits.append([])
            train_splits.append(ids)
        elif train_share <= 0:
            # test, train = ids, []
            test_splits.append(ids)
            train_splits.append([])
        elif n_splits == 1:
            _train, _test = train_test_split(
                ids,
                test_size=math.ceil(len(ids) * train_share),
                shuffle=True,
                stratify=labels
            )
            test_splits.append(_test)
            train_splits.append(_train)
        else:
            stratifier = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            folds = stratifier.split(ids, labels)
            for f in folds:
                test_splits.append(f[1])
                train_splits.append(f[0])

        for fold_id in range(len(test_splits)):
            train_splits[fold_id], split_train_embeddings, split_train_labels, split_train_names = BSA.augment_train(
                labels,
                names,
                train_splits[fold_id],
                test_splits[fold_id],
                augmented_embeddings_mapping_to_original,
                embeddings,
                all_aug_embeddings
            )

            split_test_labels, split_test_names, split_test, split_test_embeddings = BSA.augment_test(labels, names, test_splits[fold_id], embeddings)
            ###################################################################################
            # split statistics calculation:
            # lets count number of test items in each class
            test_cls_counter = {0: 0, 1: 0, 3: 0}
            train_cls_counter = {0: 0, 1: 0, 3: 0}

            for id in split_test:
                test_cls_counter[labels[id]] += 1

            for id in train_splits[fold_id]:
                train_cls_counter[split_train_labels[id]] += 1
            ####################################################################################
            print(f"DS fold {fold_id}:  Train {len(train_splits[fold_id])} : {train_cls_counter}, Test {len(split_test)} : {test_cls_counter}")
            datasets.append(
                (BSA(
                    split_train_embeddings,
                    split_train_labels,
                    split_train_names,
                    train_splits[fold_id],
                    legacy_preds
                ),
                BSA(
                    split_test_embeddings,
                    split_test_labels,
                    split_test_names,
                    split_test,
                    legacy_preds
                )
                )
            )
        # print("Train/Test loaders assembled...")
        return datasets

    def __len__(self):

        return len(self.seq_to_idx)

    def __getitem__(self, index):

        id = self.seq_to_idx[index]
        return self.embeddings[id], self.labels[id], self.names[id]#, self.legacy_preds[id]

    def get_classes_number(self):
        return len(self.classes)

    def get_classes(self):
        return self.classes

    def get_class_name(self, id):
        return self.classes[id]

    def get_embeddings_size(self):
        return self.embeddings[0].shape[0]


def accuracy_by_class(classed_names, y_true, y_pred):
    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}
    # Calculate the accuracy for each one of our classes
    # print("unique true classes", np.unique(y_true))
    preds_aggregated = Counter(y_true)
    for idx, cls in enumerate(np.unique(y_true)):
        # True negatives are all the samples that are not our current GT class (not the current row)
        # and were not predicted as the current class (not the current column)
        # true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))

        # True positives are all the samples of our current GT class that were predicted as such
        true_positives = cm[idx, idx]

        # The accuracy for the current class is the ratio between correct predictions to all predictions
        per_class_accuracies[classed_names[cls]] = round(100 * true_positives / preds_aggregated[cls], 2)

    per_class_accuracies['avg'] = round(100 * accuracy_score(y_true, y_pred), 2)
    return per_class_accuracies


def train_epoch(model, loader, criterion, classes_names, optimizer):
    model.train()
    running_loss = 0.0
    processed_size = 0
    optimizer.zero_grad()
    all_true_labels = np.array([], dtype='uint16')
    all_preds = np.array([], dtype='uint16')
    for inputs, labels, _ in loader:
        all_true_labels = np.hstack((all_true_labels, labels))

        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        processed_size += inputs.size(0)
        optimizer.zero_grad()
        all_preds = np.hstack((all_preds, preds.detach().cpu()))

    train_loss = running_loss / processed_size
    acc = accuracy_by_class(classes_names, all_true_labels, all_preds)

    return train_loss, acc


def eval_epoch(model, val_loader, criterion, classes_names):
    model.eval()
    running_loss = 0.0
    processed_size = 0

    all_true_labels = np.array([], dtype='uint16')
    all_preds = np.array([], dtype='uint16')
    all_img_names = np.array([])
    for inputs, labels, image_names in val_loader:
        all_true_labels = np.hstack((all_true_labels, labels))

        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        processed_size += inputs.size(0)
        all_preds = np.hstack((all_preds, preds.cpu()))
        all_img_names = np.hstack((all_img_names, image_names))

    val_loss = running_loss / processed_size
    # val_acc = running_corrects.cpu().numpy() / processed_size
    f1b = f1_score(all_true_labels, all_preds, average='weighted')

    return val_loss, accuracy_by_class(classes_names, all_true_labels, all_preds), f1b, confusion_matrix(
        all_true_labels, all_preds), confusion_matrix_with_files(all_true_labels, all_preds, all_img_names)


def predict_classes(model, val_loader):
    model.eval()
    all_preds = np.array([], dtype='uint16')
    all_img_names = []
    for inputs, _, img_names in val_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        all_img_names.extend(img_names)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)

        all_preds = np.append(all_preds, preds.cpu().numpy().astype('int16'))

    return all_img_names, all_preds


def confusion_matrix_with_files(preds, true_labels, img_names):
    confusion_idxs = np.where(preds != true_labels)
    cm = {}

    for idx in confusion_idxs[0]:
        key = str(true_labels[idx]) + "_" + str(preds[idx])
        if key in cm:
            cm[key].append(img_names[idx])
        else:
            cm[key] = [img_names[idx]]

    # print(cm)
    return cm


def train(train_loader, val_loader, model, epochs, classes_names, lr, supress_output=False, no_improvement_epochs_to_stop=5):
    best_accuracy = 0
    best_loss = np.inf
    best_loss_cfm = None
    best_loss_cfm_file_names = None
    best_model_weights = None
    epochs_with_no_improvement = 0
    test_loss = 0.0
    test_acc = 0.0
    best_f1 = 0
    f1 = 0
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=1, min_lr=0.00001)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, classes_names, opt)
        if val_loader:
            test_loss, test_acc, f1, cfm, cfm_file_names = eval_epoch(model, val_loader, criterion, classes_names, )

            if f1 >= best_f1: # test_loss <= best_loss:
                best_loss = test_loss
                best_loss_cfm = cfm
                best_loss_cfm_file_names = cfm_file_names
                epochs_with_no_improvement = 0
                best_f1 = f1
                best_accuracy = test_acc
            elif best_f1 >= 0.934:
                epochs_with_no_improvement += 1

            scheduler.step(test_loss)
            if epoch != 0:
                lr = scheduler._last_lr[-1]

        elif epoch == epochs - 1:
            best_model_weights = copy.deepcopy(model.state_dict())

        if not supress_output:
            print(
                f"Epoch {epoch:02d} lr {lr:0.6f} train_loss {train_loss:0.3f} train_acc {train_acc} test_loss {test_loss:0.3f} test_acc {test_acc} F1 {f1:0.3f} T:{((time.time() - epoch_start_time) / 60):0.1f}")
        
        if epochs_with_no_improvement == no_improvement_epochs_to_stop:
            break

    return best_accuracy, best_loss, best_f1, best_model_weights, best_loss_cfm, best_loss_cfm_file_names


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    DEVICE = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("Training on CPU")

    train_share = .85 if len(sys.argv) < 3 else float(sys.argv[2])
    n_epochs = 100 if len(sys.argv) < 4 else int(sys.argv[3])
    n_iterations = 1 if len(sys.argv) < 5 else int(sys.argv[4])
    lr = 0.00005 if len(sys.argv) < 6 else float(sys.argv[5]) # 0.0001 is faster
    report_confusion_files = False if len(sys.argv) < 7 else True
    verbose = True if len(sys.argv) < 8 else False

    if train_share >= 1:
        n_splits = 1
    else:
        n_splits = int(1.0 / (1.0 - train_share))

    splitted_folded_datasets = BSA.train_test_split(sys.argv[1], train_share=train_share, n_splits=n_splits)
    global_accuracy = {cls: [] for cls in splitted_folded_datasets[0][0].get_classes()}
    global_f1s = []

    for iteration in range(n_iterations):
        iteration_accuracy = {cls: [] for cls in global_accuracy.keys()}
        iteration_f1s = []

        for id, dataset in enumerate(splitted_folded_datasets):
            train_loader = DataLoader(
                dataset=dataset[0],
                batch_size=len(dataset[0]),
                num_workers=0,
                shuffle=False,
                collate_fn=None,
                pin_memory=True,
                pin_memory_device='cuda:0'
            )
            if train_share != 1.0:
                test_loader = DataLoader(
                    dataset=dataset[1],
                    batch_size=len(dataset[1]),
                    num_workers=0,
                    shuffle=False,
                    collate_fn=None,
                    pin_memory=True,
                    pin_memory_device='cuda:0',
                )
            else:
                test_loader = None

            model = PDClassifier(dataset[0].get_embeddings_size(), dataset[0].get_classes_number()).to(DEVICE)
            fold_accuracy, _, fold_f1, model_weights, _, cfm_file_names = train(
                train_loader,
                test_loader,
                model,
                n_epochs,
                dataset[0].classes,
                lr,
                supress_output=not verbose
            )

            if train_share == 1.0:
                torch.save(model_weights, os.path.join(sys.argv[1], MODEL_FN))
                sys.exit(0)

            for cls in iteration_accuracy.keys():
                iteration_accuracy[cls].append(fold_accuracy[cls])
            iteration_f1s.append(fold_f1)
            print(f"Iteration {iteration}, Fold {id} Acc:{fold_accuracy}, F1:{fold_f1:.3f}")
            if report_confusion_files:
                print(cfm_file_names)

        global_f1s.append(np.mean(iteration_f1s))
        for cls in global_accuracy.keys():
            global_accuracy[cls].append(np.mean(iteration_accuracy[cls]))

        message_accuracy = [f"{cls}: {np.mean(iteration_accuracy[cls]):0.2f}%" for cls in iteration_accuracy.keys()]
        print(f"Iteration {iteration}: Accuracy by class:{message_accuracy} F1:{global_f1s[-1]:0.3f}")

    message_accuracy = [f"{cls}: {np.mean(global_accuracy[cls]):0.2f}%" for cls in global_accuracy.keys()]
    print(f"Averaged: Accuracy by class: {message_accuracy} F1:{np.mean(global_f1s):0.3f}")
