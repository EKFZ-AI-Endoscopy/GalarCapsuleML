import argparse
from sklearn.calibration import LabelEncoder
import glob
import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import questionary


def load_videos(label_dir_base:str, technical, test_videos:bool=False) -> pd.DataFrame:
    """Load label csv files and concatenate all labels into one pd.DataFrame."""

    # if technical is selected, only use the videos 5,7,9,13,22,14 which are annotated for these labels

    label_dir = os.path.join(label_dir_base, '*')

    csv_files = glob.glob(label_dir)

    csv_files = [os.path.basename(f).split('.')[0] for f in csv_files if len(os.path.basename(f)) <= 8]

    print('Read {} csv annotation files.'.format(len(csv_files)))

    df = None

    for csv_index in csv_files:
        if technical is not None and technical is not False:
            print("Use technical video ids.")
            technical_videos = [5,8,9,13,22,14]
            if int(csv_index) not in technical_videos:
                continue

        elif test_videos:
            if int(csv_index) < 61:
                continue
        else:
            if int(csv_index) >= 61:
                continue

        tmp_df = pd.read_csv(os.path.join(label_dir_base, '{}.csv'.format(csv_index)))
        tmp_df['video'] = int(csv_index)
        if df is None:
            df = tmp_df
        else:
            df = pd.concat([df, tmp_df])

    df.reset_index(drop=True, inplace=True)

    return df


def get_label_combination(y:list):
    # encode all combinations of labels for multi-label problem # this is a very naive approach to balance the classes when using StratifiedGroupKFold - this is no good solution to this problem!
    return LabelEncoder().fit_transform([''.join(str(l)) for l in y])

def main(args) -> None:

    if args.problem == 'section':
        labels = ['mouth', 'esophagus', 'stomach', 'small intestine', 'colon']
    elif args.problem == 'technical_multiclass':
        labels = ['good view', 'reduced view', 'no view']
    elif args.problem == 'technical_multilabel':
        print('Need to filter technical frames.')
        labels = ['dirt', 'bubbles']
    elif args.problem in ['z-line', 'pylorus', 'ampulla of vater', 'ileocecal valve', 'ulcer', 'polyp', 'active bleeding', 'blood', 'erythema', 'erosion', 'angiectasia']:
        # just a single label
        labels = [args.problem]
    else:
        print('Classification problem not known!')
        exit()

    problem_save_path = os.path.join(args.split_dir, args.problem)

    if os.path.exists(problem_save_path):
        print(f'Problem Path already exists: {problem_save_path}')
    else:
        print(f'Create path: {problem_save_path}')
        os.makedirs(problem_save_path)

    if args.problem == 'technical_multiclass' or args.problem == 'technical_multilabel':
        df = load_videos(args.label_dir, technical=args.problem)
        if args.problem == 'technical_multiclass':
            df.loc[(df['reduced view'] == 1) & df['no view'] == 1, 'reduced view'] = 0
            df['good view'] = 1 - df['reduced view'] - df['no view']
    else:
        df = load_videos(args.label_dir, technical=False, test_videos=False)
        df_test = load_videos(args.label_dir, technical=False, test_videos=True)

    df_labels = df[labels]
    y = get_label_combination(df_labels.values.tolist())
    groups = list(df['video'])

    X = list(range(0,len(df_labels)))

    print('GROUPS: {}'.format(set(groups)))

    if args.problem == 'technical_multiclass' or args.problem == 'technical_multilabel':
        sgkf_test = StratifiedGroupKFold(n_splits=args.num_folds+1, shuffle=True, random_state=69)
        sgkf_trainval = StratifiedGroupKFold(n_splits=args.num_folds, shuffle=False)

        trainval_index, test_index = next(sgkf_test.split(X,y,groups))

        test_split_path = os.path.join(problem_save_path, 'test.csv')

        df_trainval = df[df.index.isin(list(trainval_index))]
        df_test = df[df.index.isin(list(test_index))]

        for label in labels:
            print(f'\nLabel: {label} TrainVal: {df_trainval[label].value_counts()} Test: {df_test[label].value_counts()}')

        column_names = labels.copy()
        column_names.append('frame')
        column_names.append('video')

        df_test = df_test[column_names]

        df_test['path'] = df_test['video'].astype(str).str.zfill(3)+  '/frame_' + df_test['frame'].astype(str).str.zfill(6) + '.PNG'

        column_names.remove('frame')
        column_names.remove('video')
        column_names.append('path')

        df_test = df_test[column_names]

        df_test.to_csv(test_split_path, index=False)

        df_labels = df_trainval[labels]
        y = get_label_combination(df_labels.values.tolist())
        groups = list(df_trainval['video'])

        print(f'GROUPS TRAINVAL: {set(groups)}')

        X = list(range(0,len(df_labels)))

        for i, (train_index, val_index) in enumerate(sgkf_trainval.split(X, y, groups)):

            split_path = os.path.join(problem_save_path, f'split_{i}')

            train_split_path = os.path.join(split_path, 'train.csv')
            val_split_path = os.path.join(split_path, 'val.csv')

            if os.path.exists(split_path) and os.path.isfile(train_split_path) and os.path.isfile(val_split_path):
                if not questionary.confirm('The split files are already in existence! Do you want to continue? THE EXISTING FILES WILL BE OVERWRITTEN!').ask():
                    exit()
            else:
                os.makedirs(split_path)

            df_train = df_trainval[df_trainval.index.isin(list(train_index))]
            df_val = df_trainval[df_trainval.index.isin(list(val_index))]

            for label in labels:
                print(f'\nLabel: {label}')
                print(df_train[label].value_counts())
                print(df_val[label].value_counts())

            column_names = labels.copy()
            column_names.append('frame')
            column_names.append('video')

            df_train = df_train[column_names]
            df_val = df_val[column_names]

            df_train['path'] = df_train['video'].astype(str).str.zfill(3)+  '/frame_' + df_train['frame'].astype(str).str.zfill(6) + '.PNG'
            df_val['path'] = df_val['video'].astype(str).str.zfill(3)+  '/frame_' + df_val['frame'].astype(str).str.zfill(6) + '.PNG'

            column_names.remove('frame')
            column_names.remove('video')
            column_names.append('path')

            df_train = df_train[column_names]
            df_val = df_val[column_names]

            df_train.to_csv(train_split_path, index=False)
            df_val.to_csv(val_split_path, index=False)

    else:
        sgkf_trainval = StratifiedGroupKFold(n_splits=args.num_folds, shuffle=False)

        test_split_path = os.path.join(problem_save_path, 'test.csv')

        column_names = labels.copy()
        column_names.append('frame')
        column_names.append('video')

        df_test = df_test[column_names]

        df_test['path'] = df_test['video'].astype(str).str.zfill(3)+  '/frame_' + df_test['frame'].astype(str).str.zfill(6) + '.PNG'

        column_names.remove('frame')
        column_names.remove('video')
        column_names.append('path')

        df_test = df_test[column_names]

        df_test.to_csv(test_split_path, index=False)

        for i, (train_index, val_index) in enumerate(sgkf_trainval.split(X, y, groups)):

            split_path = os.path.join(problem_save_path, f'split_{i}')

            train_split_path = os.path.join(split_path, 'train.csv')
            val_split_path = os.path.join(split_path, 'val.csv')

            if os.path.exists(split_path) and os.path.isfile(train_split_path) and os.path.isfile(val_split_path):
                if not questionary.confirm('The split files are already in existence! Do you want to continue? THE EXISTING FILES WILL BE OVERWRITTEN!').ask():
                    exit()
            else:
                os.makedirs(split_path)

            df_train = df[df.index.isin(list(train_index))]
            df_val = df[df.index.isin(list(val_index))]
            
            column_names = labels.copy()
            column_names.append('frame')
            column_names.append('video')

            df_train = df_train[column_names]
            df_val = df_val[column_names]

            df_train['path'] = df_train['video'].astype(str).str.zfill(3)+  '/frame_' + df_train['frame'].astype(str).str.zfill(6) + '.PNG'
            df_val['path'] = df_val['video'].astype(str).str.zfill(3)+  '/frame_' + df_val['frame'].astype(str).str.zfill(6) + '.PNG'

            column_names.remove('frame')
            column_names.remove('video')
            column_names.append('path')

            df_train = df_train[column_names]
            df_val = df_val[column_names]

            df_train.to_csv(train_split_path, index=False)
            df_val.to_csv(val_split_path, index=False)


if __name__ == "__main__":
    """Generate data splits for k-fold cross-validation. As we need to split patient-/video-wise into the different test-sets, keeping class balance here is a bit more challenging."""

    parser = argparse.ArgumentParser(description='Set params for CVAT image extraction.')

    parser.add_argument('--label_dir', type=str, required=True, help='Path where label csv files are saved as xxx.csv')
    parser.add_argument('--split_dir', type=str, required=True, help='Where to save the split csv files to.')
    parser.add_argument('--problem', type=str, required=True, help='Classification type. e.g. section, pathologial_frequent', choices=['section', 'others', 'technical_multiclass', 'technical_multilabel', 'blood'])
    parser.add_argument('--test_set', type=int, default=1)
    parser.add_argument('--num_folds', type=int, default=5)

    args = parser.parse_args()

    if args.problem == 'others':
        labels = ['z-line', 'pylorus', 'ampulla of vater', 'ileocecal valve', 'ulcer', 'polyp', 'active bleeding', 'blood', 'erythema', 'erosion', 'angiectasia']

        for label in labels:
            args.problem = label
            main(args)
    else:
        main(args)